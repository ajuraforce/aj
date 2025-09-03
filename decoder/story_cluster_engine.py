import re
from collections import defaultdict

# Safe numpy import for clustering
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Fallback numpy functions
    class np:
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0.0
import sqlite3
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class StoryClusterEngine:
    def __init__(self):
        self.db_path = "patterns.db"
        
        # Asset mapping for cross-asset impact
        self.asset_categories = {
            'oil_energy': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'CRUDE_OIL'],
            'crypto': ['BTC', 'ETH', 'ADA', 'SOL'],
            'banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS'],
            'tech': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'TECHM.NS'],
            'pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS'],
            'auto': ['MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS'],
            'metals': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'SAIL.NS']
        }
        
        # Initialize database tables
        self.init_database()
    
    def init_database(self):
        """Initialize news tables with narrative support"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # News articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT,
                link TEXT,
                source TEXT,
                published DATETIME,
                content TEXT,
                sentiment_score REAL DEFAULT 0.0,
                relevance_score REAL DEFAULT 0.0,
                mentioned_assets TEXT,
                sector TEXT,
                urgency TEXT DEFAULT 'low',
                narrative_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # News narratives table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_narratives (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                theme TEXT,
                description TEXT,
                heat_score REAL DEFAULT 0.0,
                cross_source_count INTEGER DEFAULT 1,
                affected_assets TEXT,
                impact_level TEXT DEFAULT 'low',
                time_decay_factor REAL DEFAULT 1.0,
                priority_score REAL DEFAULT 0.0,
                first_seen DATETIME,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                article_count INTEGER DEFAULT 0
            )
        ''')
        
        # Asset impact mapping
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS narrative_asset_impact (
                narrative_id TEXT,
                asset_symbol TEXT,
                impact_type TEXT,
                impact_score REAL,
                FOREIGN KEY (narrative_id) REFERENCES news_narratives(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("News database schema initialized")
    
    async def identify_story_theme(self, articles: List[Dict]) -> Dict:
        """Use AI to identify common theme among articles"""
        try:
            # Check if OpenAI is available
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                return self.fallback_theme_identification(articles)
            
            # Import OpenAI only if needed
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
            except ImportError:
                logger.warning("OpenAI library not available, using fallback theme identification")
                return self.fallback_theme_identification(articles)
            
            # Combine titles and summaries
            combined_text = "\n".join([
                f"Source: {article['source']} - {article['title']}: {article.get('summary', '')[:200]}"
                for article in articles
            ])
            
            prompt = f"""
            Analyze these related news articles and identify the main story theme:
            
            {combined_text}
            
            Provide a JSON response with:
            - narrative_title: concise story headline (max 80 chars)
            - theme: category (oil_crisis, crypto_regulation, geopolitical_tension, earnings, etc.)
            - description: 2-sentence summary of the story
            - key_assets: list of likely affected assets/sectors
            - impact_level: "high", "medium", or "low"
            """
            
            response = await client.chat.completions.acreate(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Theme identification error: {e}")
            return self.fallback_theme_identification(articles)
    
    def fallback_theme_identification(self, articles: List[Dict]) -> Dict:
        """Fallback theme identification without AI"""
        # Simple keyword-based categorization
        all_text = " ".join([f"{article['title']} {article.get('summary', '')}" for article in articles]).lower()
        
        theme_keywords = {
            'crypto_regulation': ['crypto', 'bitcoin', 'blockchain', 'regulation', 'sec', 'ban'],
            'oil_crisis': ['oil', 'crude', 'energy', 'petroleum', 'opec', 'gas'],
            'banking_policy': ['bank', 'interest', 'rate', 'fed', 'rbi', 'monetary', 'policy'],
            'tech_earnings': ['tech', 'technology', 'earnings', 'revenue', 'profit', 'growth'],
            'geopolitical_tension': ['war', 'conflict', 'tension', 'sanctions', 'trade', 'diplomatic']
        }
        
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                theme_scores[theme] = score
        
        # Determine best theme
        if theme_scores:
            best_theme = max(theme_scores, key=theme_scores.get)
        else:
            best_theme = 'general'
        
        # Extract potential assets
        asset_keywords = []
        for category, assets in self.asset_categories.items():
            for asset in assets:
                if asset.lower().replace('.ns', '') in all_text:
                    asset_keywords.append(asset)
        
        # Determine impact level based on keyword frequency
        keyword_count = sum(theme_scores.values())
        if keyword_count >= 5:
            impact_level = 'high'
        elif keyword_count >= 3:
            impact_level = 'medium'
        else:
            impact_level = 'low'
        
        return {
            'narrative_title': f"Story from {len(articles)} sources - {best_theme.replace('_', ' ').title()}",
            'theme': best_theme,
            'description': f"Multiple news sources reporting on {best_theme.replace('_', ' ')} developments. Related articles from {len(set(article['source'] for article in articles))} sources.",
            'key_assets': asset_keywords[:5],  # Limit to 5 assets
            'impact_level': impact_level
        }
    
    def calculate_similarity(self, article1: Dict, article2: Dict) -> float:
        """Calculate similarity between two articles"""
        text1 = f"{article1['title']} {article1.get('summary', '')}".lower()
        text2 = f"{article2['title']} {article2.get('summary', '')}".lower()
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def group_similar_articles(self, articles: List[Dict], similarity_threshold: float = 0.3) -> List[List[Dict]]:
        """Group articles by similarity"""
        clusters = []
        used_articles = set()
        
        for i, article in enumerate(articles):
            if article['id'] in used_articles:
                continue
                
            cluster = [article]
            used_articles.add(article['id'])
            
            for j, other_article in enumerate(articles[i+1:], i+1):
                if other_article['id'] in used_articles:
                    continue
                    
                similarity = self.calculate_similarity(article, other_article)
                if similarity >= similarity_threshold:
                    cluster.append(other_article)
                    used_articles.add(other_article['id'])
            
            clusters.append(cluster)
        
        return clusters
    
    def map_assets_to_narrative(self, theme: str, key_assets: List[str]) -> List[Dict]:
        """Map narrative to trading assets"""
        asset_impacts = []
        
        # Theme-based asset mapping
        theme_mappings = {
            'oil_crisis': self.asset_categories['oil_energy'],
            'crypto_regulation': self.asset_categories['crypto'],
            'banking_policy': self.asset_categories['banking'],
            'tech_earnings': self.asset_categories['tech']
        }
        
        mapped_assets = theme_mappings.get(theme, [])
        
        # Add explicitly mentioned assets
        for asset in key_assets:
            asset_upper = asset.upper()
            if asset_upper not in mapped_assets:
                mapped_assets.append(asset_upper)
        
        # Create impact records
        for asset in mapped_assets:
            impact_score = 0.8 if asset in key_assets else 0.6
            asset_impacts.append({
                'asset_symbol': asset,
                'impact_type': 'price_movement',
                'impact_score': impact_score
            })
        
        return asset_impacts
    
    def calculate_priority_score(self, cross_source_count: int, impact_level: str, time_decay: float) -> float:
        """Calculate overall priority score"""
        # Cross-source multiplier (more sources = higher priority)
        source_multiplier = min(cross_source_count / 3, 2.0)  # Cap at 2x
        
        # Impact level multiplier
        impact_multiplier = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(impact_level, 0.4)
        
        # Time decay (recent = higher priority)
        time_multiplier = time_decay
        
        return source_multiplier * impact_multiplier * time_multiplier
    
    def calculate_time_decay(self, cluster: List[Dict]) -> float:
        """Calculate time decay factor for cluster"""
        now = datetime.now()
        
        # Get the most recent article time
        most_recent = max([
            datetime.fromisoformat(article['published'].replace('Z', ''))
            for article in cluster
            if article.get('published')
        ], default=now)
        
        # Calculate hours since most recent article
        hours_since = (now - most_recent).total_seconds() / 3600
        
        # Exponential decay (half-life of 6 hours)
        return max(0.1, 2 ** (-hours_since / 6))
    
    async def create_narratives_from_recent_articles(self, hours_back: int = 24):
        """Process recent articles into narratives"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent unprocessed articles
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        cursor.execute('''
            SELECT id, title, summary, link, source, published, mentioned_assets
            FROM news_articles 
            WHERE published > ? AND (narrative_id IS NULL OR narrative_id = '')
            ORDER BY published DESC
        ''', (cutoff_time,))
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                'id': row[0], 'title': row[1], 'summary': row[2], 'link': row[3],
                'source': row[4], 'published': row[5], 'mentioned_assets': row[6]
            })
        
        if not articles:
            logger.info("No new articles to process")
            return 0
        
        # Group similar articles
        clusters = self.group_similar_articles(articles, similarity_threshold=0.3)
        
        narratives_created = 0
        
        for cluster in clusters:
            if len(cluster) < 2:  # Skip single-article "narratives"
                continue
            
            try:
                # Identify theme and create narrative
                theme_data = await self.identify_story_theme(cluster)
                
                # Generate narrative ID
                narrative_id = hashlib.md5(f"{theme_data['narrative_title']}{datetime.now()}".encode()).hexdigest()
                
                # Calculate metrics
                cross_source_count = len(set(article['source'] for article in cluster))
                time_decay = self.calculate_time_decay(cluster)
                heat_score = cross_source_count * len(cluster) * 0.1
                priority_score = self.calculate_priority_score(cross_source_count, theme_data['impact_level'], time_decay)
                
                # Map to assets
                asset_impacts = self.map_assets_to_narrative(theme_data['theme'], theme_data['key_assets'])
                
                # Insert narrative
                cursor.execute('''
                    INSERT INTO news_narratives 
                    (id, title, theme, description, heat_score, cross_source_count, 
                     affected_assets, impact_level, time_decay_factor, priority_score, 
                     first_seen, article_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    narrative_id, theme_data['narrative_title'], theme_data['theme'],
                    theme_data['description'], heat_score, cross_source_count,
                    ','.join(theme_data['key_assets']), theme_data['impact_level'],
                    time_decay, priority_score, datetime.now(), len(cluster)
                ))
                
                # Update articles with narrative_id
                for article in cluster:
                    cursor.execute('UPDATE news_articles SET narrative_id = ? WHERE id = ?', 
                                 (narrative_id, article['id']))
                
                # Insert asset impacts
                for impact in asset_impacts:
                    cursor.execute('''
                        INSERT INTO narrative_asset_impact 
                        (narrative_id, asset_symbol, impact_type, impact_score)
                        VALUES (?, ?, ?, ?)
                    ''', (narrative_id, impact['asset_symbol'], impact['impact_type'], impact['impact_score']))
                
                narratives_created += 1
                logger.info(f"Created narrative: {theme_data['narrative_title']}")
                
            except Exception as e:
                logger.error(f"Error creating narrative: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created {narratives_created} narratives from {len(articles)} articles")
        return narratives_created
    
    def store_news_article(self, article_data: Dict):
        """Store individual news article in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO news_articles 
                (id, title, summary, link, source, published, content, sentiment_score, 
                 relevance_score, mentioned_assets, sector, urgency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article_data['id'],
                article_data['title'],
                article_data.get('summary', ''),
                article_data.get('link', ''),
                article_data.get('source', ''),
                article_data.get('published', datetime.now()),
                article_data.get('content', ''),
                article_data.get('sentiment_score', 0.0),
                article_data.get('relevance_score', 0.0),
                ','.join(article_data.get('mentioned_assets', [])),
                article_data.get('sector', ''),
                article_data.get('urgency', 'low')
            ))
            
            conn.commit()
            logger.debug(f"Stored article: {article_data['id']}")
            
        except Exception as e:
            logger.error(f"Error storing article: {e}")
        finally:
            conn.close()
    
    def get_recent_narratives(self, limit: int = 10) -> List[Dict]:
        """Get recent narratives for API consumption"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT n.id, n.title, n.description, n.heat_score, n.impact_level, 
                   n.priority_score, n.cross_source_count, n.first_seen, n.affected_assets,
                   COUNT(a.id) as article_count
            FROM news_narratives n
            LEFT JOIN news_articles a ON n.id = a.narrative_id
            GROUP BY n.id
            ORDER BY n.priority_score DESC, n.heat_score DESC
            LIMIT ?
        ''', (limit,))
        
        narratives = []
        for row in cursor.fetchall():
            narratives.append({
                'id': row[0], 'title': row[1], 'description': row[2], 'heat_score': row[3],
                'impact_level': row[4], 'priority_score': row[5], 'cross_source_count': row[6],
                'first_seen': row[7], 'affected_assets': row[8], 'article_count': row[9]
            })
        
        conn.close()
        return narratives

    def cluster_articles_into_cards(self, articles: List[Dict]) -> List[Dict]:
        """Enhanced clustering to create topic-based cards with TF-IDF"""
        try:
            if not articles:
                return []
            
            # Try using TF-IDF clustering if sklearn is available
            if self._can_use_advanced_clustering():
                return self._advanced_tfidf_clustering(articles)
            else:
                return self._fallback_keyword_clustering(articles)
                
        except Exception as e:
            logger.error(f"Error in article clustering: {e}")
            return self._fallback_keyword_clustering(articles)
    
    def _can_use_advanced_clustering(self) -> bool:
        """Check if advanced clustering dependencies are available"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            import numpy as np
            return True
        except ImportError:
            return False
    
    def _advanced_tfidf_clustering(self, articles: List[Dict]) -> List[Dict]:
        """Advanced clustering using TF-IDF and KMeans"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Prepare text data for clustering
            texts = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('summary', '')}"
                texts.append(text)
            
            if len(texts) < 2:
                return self._create_single_card(articles)
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Determine optimal number of clusters (max 5)
            n_clusters = min(max(2, len(articles) // 3), 5)
            
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group articles by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[label].append(articles[idx])
            
            # Create cards from clusters
            cards = []
            feature_names = vectorizer.get_feature_names_out()
            
            for cluster_id, cluster_articles in clusters.items():
                if len(cluster_articles) < 1:
                    continue
                
                # Extract top terms for this cluster
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_tfidf = tfidf_matrix[cluster_indices]
                mean_tfidf = np.mean(cluster_tfidf, axis=0).A1
                
                # Get top 3 terms
                top_indices = mean_tfidf.argsort()[-3:][::-1]
                top_terms = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]
                
                # Create card
                card = self._create_card_from_cluster(cluster_articles, top_terms)
                cards.append(card)
            
            return sorted(cards, key=lambda x: x['priority_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Advanced clustering failed: {e}")
            return self._fallback_keyword_clustering(articles)
    
    def _fallback_keyword_clustering(self, articles: List[Dict]) -> List[Dict]:
        """Fallback clustering using keyword matching"""
        try:
            # Define topic keywords
            topic_keywords = {
                'crypto': ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'btc', 'eth', 'defi', 'nft'],
                'markets': ['market', 'stock', 'trading', 'price', 'rally', 'crash', 'bull', 'bear'],
                'tech': ['technology', 'tech', 'ai', 'artificial intelligence', 'startup', 'innovation'],
                'economy': ['economy', 'inflation', 'fed', 'interest rate', 'gdp', 'recession'],
                'geopolitics': ['war', 'conflict', 'sanctions', 'government', 'policy', 'election']
            }
            
            # Group articles by topics
            topic_clusters = defaultdict(list)
            unassigned = []
            
            for article in articles:
                text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
                assigned = False
                
                for topic, keywords in topic_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        topic_clusters[topic].append(article)
                        assigned = True
                        break
                
                if not assigned:
                    unassigned.append(article)
            
            # Create cards
            cards = []
            
            for topic, cluster_articles in topic_clusters.items():
                if cluster_articles:
                    card = self._create_card_from_cluster(cluster_articles, [topic.title()])
                    cards.append(card)
            
            # Handle unassigned articles
            if unassigned:
                card = self._create_card_from_cluster(unassigned, ['General News'])
                cards.append(card)
            
            return sorted(cards, key=lambda x: x['priority_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Fallback clustering failed: {e}")
            return self._create_single_card(articles)
    
    def _create_card_from_cluster(self, articles: List[Dict], key_terms: List[str]) -> Dict:
        """Create a card from a cluster of articles"""
        if not articles:
            return {}
        
        # Calculate metrics
        avg_sentiment = np.mean([article.get('sentiment_score', 0.0) for article in articles]) if articles else 0.0
        total_articles = len(articles)
        
        # Get unique sources
        sources = list(set(article.get('source', 'Unknown') for article in articles))
        cross_source_count = len(sources)
        
        # Calculate priority score
        priority_score = min(1.0, (cross_source_count * 0.3) + (total_articles * 0.1) + (abs(avg_sentiment) * 0.5))
        
        # Determine impact level
        if priority_score > 0.7:
            impact_level = 'high'
        elif priority_score > 0.4:
            impact_level = 'medium'
        else:
            impact_level = 'low'
        
        # Create headline from key terms
        headline = ', '.join(key_terms[:3]) if key_terms else 'Market News'
        
        # Get top 3 articles for display
        top_articles = sorted(articles, key=lambda x: x.get('sentiment_score', 0.0), reverse=True)[:3]
        
        return {
            'headline': headline,
            'articles': top_articles,
            'total_count': total_articles,
            'cross_source_count': cross_source_count,
            'avg_sentiment': avg_sentiment,
            'priority_score': priority_score,
            'impact_level': impact_level,
            'key_terms': key_terms
        }
    
    def _create_single_card(self, articles: List[Dict]) -> List[Dict]:
        """Create a single card when clustering fails"""
        if not articles:
            return []
        
        return [self._create_card_from_cluster(articles, ['All News'])]