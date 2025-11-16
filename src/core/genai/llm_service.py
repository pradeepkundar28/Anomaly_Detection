"""
LLM service for generating intelligent insights using Ollama with Llama models.
"""
import requests
import json
from typing import List, Dict, Optional
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLMService:
    """Service for interacting with Ollama LLM locally."""
    
    def __init__(self, 
                 model: str = "llama3.2:3b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Initialize Ollama LLM service.
        
        Args:
            model: Ollama model name (e.g., 'llama3.2:3b', 'llama2', 'mistral')
            base_url: Ollama API base URL
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Check if Ollama is available
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                logger.info(f"Ollama connected. Available models: {available_models}")
                
                # Check if requested model is available
                if not any(self.model in m for m in available_models):
                    logger.warning(f"Model '{self.model}' not found. Available: {available_models}")
                    logger.info(f"Run: ollama pull {self.model}")
            else:
                logger.warning(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            raise ConnectionError(f"Ollama not available at {self.base_url}. Run 'ollama serve' first.")
        
        logger.info(f"Initialized Ollama LLM service: {model}")
    
    def generate_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate completion from Ollama LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            Generated text
        """
        try:
            # Combine system and user prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Call Ollama API
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # 2 minutes timeout for generation
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"[Error: Ollama API returned {response.status_code}]"
                
        except requests.exceptions.Timeout:
            logger.error("Ollama generation timed out")
            return "[Error: Generation timed out]"
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"[Error: {str(e)}]"
    
    def analyze_anomaly(self, anomaly_data: Dict, context_logs: List[str]) -> str:
        """
        Generate intelligent analysis of an anomaly.
        
        Args:
            anomaly_data: Anomaly details (timestamp, equipment, scores, etc.)
            context_logs: Related operator logs
            
        Returns:
            Natural language analysis
        """
        system_prompt = """You are an expert oil rig operations analyst with 20 years of experience. 
Analyze sensor anomalies and provide actionable insights about potential issues, root causes, 
and recommended actions. Be concise, technical, and prioritize safety."""
        
        # Prepare log context
        log_context = "\n".join(f"  - {log}" for log in context_logs[:5]) if context_logs else "  - No related logs found"
        
        prompt = f"""Analyze this anomaly detected in oil rig equipment:

**Anomaly Details:**
- Equipment: {anomaly_data.get('equipment_id', 'Unknown')}
- Timestamp: {anomaly_data.get('timestamp', 'Unknown')}
- Anomaly Score: {anomaly_data.get('anomaly_score', 0):.3f} (higher = more anomalous)
- Pressure: {anomaly_data.get('pressure', 0):.2f} PSI
- Temperature: {anomaly_data.get('temperature', 0):.2f} °C
- Vibration: {anomaly_data.get('vibration', 0):.2f} Hz

**Related Operator Logs:**
{log_context}

Provide a concise analysis covering:
1. **Root Cause Hypothesis**: What likely caused this anomaly? (2-3 sentences)
2. **Risk Assessment**: Severity level (Low/Medium/High/Critical) and potential impact
3. **Immediate Actions**: What should operators do right now? (2-3 bullet points)
4. **Prevention**: How to prevent similar issues? (1-2 sentences)

Keep the response under 200 words and be specific to oil rig operations."""
        
        return self.generate_completion(prompt, system_prompt)
    
    def generate_executive_summary(self, anomalies_df: pd.DataFrame, 
                                   logs_df: pd.DataFrame) -> str:
        """
        Generate executive summary of all anomalies.
        
        Args:
            anomalies_df: DataFrame of detected anomalies
            logs_df: DataFrame of operator logs
            
        Returns:
            Executive summary
        """
        n_anomalies = anomalies_df['is_anomaly'].sum()
        equipment_affected = anomalies_df[anomalies_df['is_anomaly']]['equipment_id'].unique().tolist()
        
        # Get top 3 most severe anomalies
        top_anomalies = anomalies_df[anomalies_df['is_anomaly']].nlargest(3, 'anomaly_score')
        
        top_anomalies_text = "\n".join(
            f"  {i+1}. {row['equipment_id']} at {row['timestamp']} (Score: {row['anomaly_score']:.3f})"
            for i, (_, row) in enumerate(top_anomalies.iterrows())
        )
        
        system_prompt = """You are a senior operations executive for an oil rig company. 
Provide high-level summaries suitable for C-level executives focused on business impact, 
risk assessment, and strategic recommendations."""
        
        prompt = f"""Generate an executive summary of anomaly detection results:

**Overview:**
- Total anomalies detected: {n_anomalies}
- Equipment affected: {', '.join(equipment_affected)}
- Time period: {anomalies_df['timestamp'].min()} to {anomalies_df['timestamp'].max()}
- Total operator logs: {len(logs_df)}

**Top 3 Most Severe Anomalies:**
{top_anomalies_text}

Provide a professional executive summary covering:
1. **Executive Summary** (2-3 sentences highlighting key concerns)
2. **Key Findings** (3-4 bullet points with specific insights)
3. **Business Impact** (risk level, potential downtime, estimated costs)
4. **Strategic Recommendations** (2-3 high-level actions for management)

Keep under 250 words, use professional business tone, focus on actionable insights."""
        
        return self.generate_completion(prompt, system_prompt)
    
    def predict_maintenance_needs(self, equipment_stats: List[Dict]) -> str:
        """
        Predict maintenance needs based on equipment patterns.
        
        Args:
            equipment_stats: List of equipment statistics with anomaly counts
            
        Returns:
            Maintenance predictions
        """
        system_prompt = """You are a predictive maintenance expert for industrial equipment. 
Based on sensor patterns and anomaly history, predict maintenance needs and prioritize actions."""
        
        stats_text = "\n".join(
            f"  - {s['equipment']}: {s['anomaly_count']} anomalies detected, "
            f"Avg Pressure: {s.get('avg_pressure', 0):.1f} PSI, "
            f"Avg Temp: {s.get('avg_temp', 0):.1f}°C, "
            f"Avg Vibration: {s.get('avg_vibration', 0):.1f} Hz"
            for s in equipment_stats
        )
        
        prompt = f"""Based on equipment sensor data and anomaly history, predict maintenance needs:

**Equipment Status:**
{stats_text}

Provide:
1. **Maintenance Priority Queue**: Which equipment needs attention first and why?
2. **Predicted Failure Timeline**: Which equipment is likely to fail soon? (days/weeks)
3. **Maintenance Type Recommendations**: Preventive, corrective, or predictive for each
4. **Cost-Benefit Analysis**: Brief assessment of maintaining now vs risk of failure

Format as an actionable maintenance schedule. Keep under 200 words."""
        
        return self.generate_completion(prompt, system_prompt)
    
    def generate_simple_summary(self, anomalies_df: pd.DataFrame) -> str:
        """
        Generate a simple summary of anomalies (lighter LLM call).
        
        Args:
            anomalies_df: DataFrame of detected anomalies
            
        Returns:
            Simple summary text
        """
        n_total = len(anomalies_df)
        n_anomalies = anomalies_df['is_anomaly'].sum()
        anomaly_rate = (n_anomalies / n_total * 100) if n_total > 0 else 0
        
        equipment_counts = anomalies_df[anomalies_df['is_anomaly']].groupby('equipment_id').size().to_dict()
        equipment_summary = ", ".join(f"{eq}: {cnt}" for eq, cnt in equipment_counts.items())
        
        system_prompt = "You are an oil rig operations analyst. Provide concise summaries."
        
        prompt = f"""Summarize these anomaly detection results in 2-3 sentences:

- Total data points: {n_total}
- Anomalies detected: {n_anomalies} ({anomaly_rate:.1f}%)
- Anomalies by equipment: {equipment_summary}

Focus on the most important insights and any concerning patterns."""
        
        return self.generate_completion(prompt, system_prompt)


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama service is available.
    
    Args:
        base_url: Ollama API base URL
        
    Returns:
        True if available, False otherwise
    """
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        print(response)
        return response.status_code == 200
    except:
        return False
