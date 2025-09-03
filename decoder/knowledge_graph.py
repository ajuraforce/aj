"""
Knowledge Graph for linking events, assets, sectors, and regimes
Uses NetworkX for graph operations with time-based decay
"""

import networkx as nx
import sqlite3
import os
import psycopg2
import json
from datetime import datetime, timedelta
import schedule
import time
import logging
from typing import Dict, List, Optional, Tuple
import pickle

logger = logging.getLogger(__name__)

class KG:
    def __init__(self, db_path='patterns.db'):
        self.graph = nx.DiGraph()
        # Use PostgreSQL if available, fallback to SQLite
        self.database_url = os.getenv('DATABASE_URL')
        if self.database_url:
            self.conn = psycopg2.connect(self.database_url)
        else:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.init_tables()
        self.load_graph()

    def init_tables(self):
        """Initialize database tables for graph persistence"""
        if self.database_url:
            # PostgreSQL
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kg_edges (
                    src_id TEXT,
                    dst_id TEXT,
                    relation TEXT,
                    weight REAL,
                    half_life_hours REAL,
                    last_updated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (src_id, dst_id, relation)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT,
                    attributes JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
        else:
            # SQLite
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS kg_edges (
                    src_id TEXT,
                    dst_id TEXT,
                    relation TEXT,
                    weight REAL,
                    half_life_hours REAL,
                    last_updated TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (src_id, dst_id, relation)
                )
            ''')
            
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT,
                    attributes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()

    def link(self, src_id: str, dst_id: str, relation: str, weight: float, half_life_hours: float):
        """Create or update link between nodes"""
        try:
            current_time = datetime.now()
            
            # Add nodes if they don't exist
            self.graph.add_node(src_id)
            self.graph.add_node(dst_id)
            
            # Add/update edge
            self.graph.add_edge(src_id, dst_id, 
                              relation=relation, 
                              weight=weight, 
                              half_life=half_life_hours,
                              last_updated=current_time)
            
            # Store in database
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO kg_edges (src_id, dst_id, relation, weight, half_life_hours, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (src_id, dst_id, relation) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    last_updated = EXCLUDED.last_updated
                ''', (src_id, dst_id, relation, weight, half_life_hours, current_time))
            else:
                self.conn.execute('''
                    INSERT OR REPLACE INTO kg_edges VALUES (?, ?, ?, ?, ?, ?)
                ''', (src_id, dst_id, relation, weight, half_life_hours, current_time.isoformat()))
            self.conn.commit()
            
            logger.info(f"Linked: {src_id} --[{relation}]--> {dst_id} (weight: {weight})")
            
        except Exception as e:
            logger.error(f"Error linking nodes: {e}")

    def neighbors(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[str, Dict]]:
        """Get neighbors of a node, optionally filtered by relation"""
        try:
            neighbors = []
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph[node_id][neighbor]
                if relation is None or edge_data.get('relation') == relation:
                    neighbors.append((neighbor, edge_data))
            return neighbors
        except:
            return []

    def find_path(self, src_id: str, dst_id: str, max_length: int = 3) -> List[str]:
        """Find shortest path between nodes"""
        try:
            return nx.shortest_path(self.graph, src_id, dst_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_influence_chain(self, event_id: str, max_hops: int = 2) -> Dict:
        """Get assets/sectors influenced by an event through graph traversal"""
        influenced = {'assets': [], 'sectors': [], 'themes': []}
        
        try:
            # BFS traversal
            queue = [(event_id, 0)]
            visited = set([event_id])
            
            while queue:
                current, hops = queue.pop(0)
                
                if hops >= max_hops:
                    continue
                
                for neighbor, edge_data in self.neighbors(current):
                    if neighbor not in visited and edge_data['weight'] > 0.3:
                        visited.add(neighbor)
                        queue.append((neighbor, hops + 1))
                        
                        # Categorize by node type
                        if neighbor.startswith('ASSET_'):
                            influenced['assets'].append({
                                'symbol': neighbor.replace('ASSET_', ''),
                                'weight': edge_data['weight'],
                                'relation': edge_data['relation']
                            })
                        elif neighbor.startswith('SECTOR_'):
                            influenced['sectors'].append({
                                'name': neighbor.replace('SECTOR_', ''),
                                'weight': edge_data['weight']
                            })
                        elif neighbor.startswith('THEME_'):
                            influenced['themes'].append({
                                'name': neighbor.replace('THEME_', ''),
                                'weight': edge_data['weight']
                            })
            
            return influenced
            
        except Exception as e:
            logger.error(f"Error getting influence chain: {e}")
            return influenced

    def decay_and_prune(self):
        """Apply time-based decay to edge weights and prune weak edges"""
        try:
            current_time = datetime.now()
            edges_to_remove = []
            updates = []
            
            for u, v, data in self.graph.edges(data=True):
                if 'last_updated' in data and 'half_life' in data:
                    # Calculate age in hours
                    if isinstance(data['last_updated'], str):
                        last_updated = datetime.fromisoformat(data['last_updated'])
                    else:
                        last_updated = data['last_updated']
                    age_hours = (current_time - last_updated).total_seconds() / 3600
                    
                    # Apply exponential decay
                    decay_factor = 0.5 ** (age_hours / data['half_life'])
                    new_weight = data['weight'] * decay_factor
                    
                    if new_weight < 0.1:  # Prune threshold
                        edges_to_remove.append((u, v))
                    else:
                        # Update weight
                        data['weight'] = new_weight
                        updates.append((u, v, data['relation'], new_weight, data['half_life'], current_time.isoformat()))
            
            # Remove weak edges
            for u, v in edges_to_remove:
                self.graph.remove_edge(u, v)
                if self.database_url:
                    cursor = self.conn.cursor()
                    cursor.execute('DELETE FROM kg_edges WHERE src_id = %s AND dst_id = %s', (u, v))
                else:
                    self.conn.execute('DELETE FROM kg_edges WHERE src_id = ? AND dst_id = ?', (u, v))
            
            # Update weights in database
            for update in updates:
                if self.database_url:
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        UPDATE kg_edges SET weight = %s, last_updated = %s 
                        WHERE src_id = %s AND dst_id = %s AND relation = %s
                    ''', (update[3], update[5], update[0], update[1], update[2]))
                else:
                    self.conn.execute('''
                        UPDATE kg_edges SET weight = ?, last_updated = ? 
                        WHERE src_id = ? AND dst_id = ? AND relation = ?
                    ''', (update[3], update[5], update[0], update[1], update[2]))
            
            self.conn.commit()
            logger.info(f"Pruned {len(edges_to_remove)} weak edges, updated {len(updates)} weights")
            
        except Exception as e:
            logger.error(f"Error in decay_and_prune: {e}")

    def strengthen_edge(self, src_id: str, dst_id: str, boost: float = 0.1):
        """Strengthen edge based on successful prediction"""
        try:
            if self.graph.has_edge(src_id, dst_id):
                current_weight = self.graph[src_id][dst_id]['weight']
                new_weight = min(1.0, current_weight + boost)
                self.graph[src_id][dst_id]['weight'] = new_weight
                
                # Update database
                if self.database_url:
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        UPDATE kg_edges SET weight = %s WHERE src_id = %s AND dst_id = %s
                    ''', (new_weight, src_id, dst_id))
                else:
                    self.conn.execute('''
                        UPDATE kg_edges SET weight = ? WHERE src_id = ? AND dst_id = ?
                    ''', (new_weight, src_id, dst_id))
                self.conn.commit()
                
                logger.info(f"Strengthened edge: {src_id} -> {dst_id} to {new_weight}")
        except Exception as e:
            logger.error(f"Error strengthening edge: {e}")

    def save_graph(self):
        """Save graph state to database"""
        try:
            # This is handled automatically through database operations
            logger.info("Graph state saved to database")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")

    def load_graph(self):
        """Load graph state from database"""
        try:
            if self.database_url:
                cursor = self.conn.cursor()
                cursor.execute('SELECT * FROM kg_edges')
            else:
                cursor = self.conn.execute('SELECT * FROM kg_edges')
                
            for row in cursor.fetchall():
                src_id, dst_id, relation, weight, half_life, last_updated = row[:6]
                self.graph.add_edge(src_id, dst_id,
                                  relation=relation,
                                  weight=weight,
                                  half_life=half_life,
                                  last_updated=last_updated)
            
            logger.info(f"Loaded graph with {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error loading graph: {e}")

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
            'density': nx.density(self.graph)
        }

# Auto-linking based on events (integrate with event normalizer)
class AutoLinker:
    def __init__(self, kg: KG):
        self.kg = kg

    def link_event_to_assets(self, event: Dict):
        """Automatically create links between events and mentioned assets"""
        event_id = f"EVENT_{event['id']}"
        assets = event.get('assets', [])
        
        for asset in assets:
            asset_id = f"ASSET_{asset}"
            
            # Link with weight based on confidence
            weight = event.get('confidence', 0.5) * 0.8  # Scale down
            
            # Different relations based on event type
            if event['type'] == 'POLICY_SIGNAL':
                relation = 'policy_affects'
                half_life = 168  # 1 week
            elif event['type'] == 'EARNINGS_SURPRISE':
                relation = 'earnings_affects'
                half_life = 72  # 3 days
            else:
                relation = 'influences'
                half_life = 48  # 2 days
            
            self.kg.link(event_id, asset_id, relation, weight, half_life)