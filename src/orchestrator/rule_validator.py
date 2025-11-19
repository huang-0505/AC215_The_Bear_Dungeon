"""
rule_validator.py

Interface to the Rule Agent for validating player actions against D&D rules.
Handles communication with the Rule Agent RAG service.
"""

from typing import Dict, Optional
import requests
import logging

logger = logging.getLogger(__name__)


class RuleValidator:
    """Interface to the Rule Agent for action validation"""

    def __init__(self, rule_agent_url: str = "http://rule-agent:9002"):
        self.rule_agent_url = rule_agent_url

    def validate_action(self, user_input: str, game_context: Dict) -> Dict:
        """
        Validate user action against D&D rules using the Rule Agent RAG system.

        Args:
            user_input: The player's action text
            game_context: Current game state context (combat state, HP, etc.)

        Returns:
            {
                "is_valid": bool,
                "validation_type": "valid" | "invalid" | "sabotage" | "no_rules",
                "rule_text": str,  # Retrieved D&D rule passages
                "explanation": str,  # Rule agent's analysis
                "suggested_correction": Optional[str]
            }
        """
        try:
            response = requests.post(
                f"{self.rule_agent_url}/validate",
                json={
                    "user_input": user_input,
                    "context": game_context
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError:
            logger.warning(f"Cannot connect to Rule Agent at {self.rule_agent_url}. Allowing action by default.")
            return {
                "is_valid": True,
                "validation_type": "no_validation",
                "rule_text": "",
                "explanation": "Rule Agent unavailable - action allowed by default"
            }

        except requests.exceptions.Timeout:
            logger.warning("Rule Agent request timed out. Allowing action by default.")
            return {
                "is_valid": True,
                "validation_type": "no_validation",
                "rule_text": "",
                "explanation": "Rule Agent timeout - action allowed by default"
            }

        except Exception as e:
            logger.error(f"Error validating action: {str(e)}")
            return {
                "is_valid": True,
                "validation_type": "error",
                "rule_text": "",
                "explanation": f"Validation error: {str(e)} - action allowed by default"
            }

    def is_sabotage(self, validation_result: Dict) -> bool:
        """Check if the input is a meta-gaming/sabotage attempt"""
        return validation_result.get("validation_type") == "sabotage"

    def get_applicable_rules(self, action: str, n_results: int = 5) -> str:
        """
        Retrieve D&D rules relevant to an action without full validation.

        Args:
            action: The action description
            n_results: Number of rule chunks to retrieve

        Returns:
            Concatenated rule text from the D&D rulebooks
        """
        try:
            response = requests.post(
                f"{self.rule_agent_url}/retrieve_rules",
                json={
                    "query": action,
                    "n_results": n_results
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("rules", "")

        except Exception as e:
            logger.error(f"Error retrieving rules: {str(e)}")
            return ""

    def check_health(self) -> bool:
        """Check if the Rule Agent service is available"""
        try:
            response = requests.get(
                f"{self.rule_agent_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
