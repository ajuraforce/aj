"""
GPT-5 Integration for event typing and narratives
"""

import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class GPTIntegrator:
    def __init__(self, api_key: str = ""):
        # Import OpenAI only if we have an API key
        self.api_key = api_key
        self.client = None
        
        if api_key and api_key.strip():
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                logger.error("OpenAI library not installed. Install with: pip install openai")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

    def type_event(self, title: str, summary: str) -> Dict:
        """
        Convert raw text to ontology event type and entities.
        """
        if not self.client:
            logger.warning("GPT integration not available - no API key or client initialization failed")
            return {
                'type': 'unknown',
                'entities': [title, summary],
                'confidence': 0.5
            }

        prompt = f"""
You are an event classifier. Given the title and summary,
output JSON with keys: type, entities (list), confidence.

Title: {title}
Summary: {summary}

Return only valid JSON.
"""
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 as GPT-5 may not be available
                messages=[{"role":"user","content":prompt}],
                max_tokens=200, 
                temperature=0
            )
            
            content = resp.choices[0].message.content
            if content:
                return json.loads(content)
            else:
                return {'type': 'empty_response', 'entities': [], 'confidence': 0}
        except json.JSONDecodeError:
            logger.error("GPT returned invalid JSON")
            return {'type': 'parsing_error', 'entities': [], 'confidence': 0}
        except Exception as e:
            logger.error(f"GPT type_event error: {e}")
            return {'type': 'error', 'entities': [], 'confidence': 0}

    def build_narrative(self, decision: Dict) -> str:
        """
        Build human-readable rationale from decision JSON.
        """
        if not self.client:
            # Fallback narrative building without GPT
            rationale = decision.get('rationale', [])
            confidence = decision.get('confidence', 0)
            action = decision.get('action', 'unknown')
            
            narrative = f"Decision: {action} with {confidence:.2f} confidence. "
            if rationale:
                narrative += "Based on: " + "; ".join(rationale[:3])  # Limit to first 3 items
            
            return narrative

        input_json = json.dumps(decision, indent=2)
        prompt = f"Create a concise narrative explaining this trading decision:\n{input_json}"
        
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 as GPT-5 may not be available
                messages=[{"role":"user","content":prompt}],
                max_tokens=100, 
                temperature=0.7
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"GPT narrative building error: {e}")
            # Fallback narrative
            return f"Trading decision: {decision.get('action', 'unknown')} with confidence {decision.get('confidence', 0):.2f}"