"""
story_tree_loader.py

Loads predefined story tree structures from JSON files.
Enables controlled story flow with guaranteed endings.
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path


class StoryNode:
    """Represents a node in a predefined story tree"""

    def __init__(
        self,
        node_id: str,
        narrative: str,
        choices: List[str],
        is_ending: bool = False,
        ending_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
        combat_available: bool = False,  # Whether combat is available at this node
    ):
        self.node_id = node_id
        self.narrative = narrative
        self.choices = choices  # List of choice text that leads to child nodes
        self.is_ending = is_ending
        self.ending_type = ending_type  # e.g., "victory", "defeat", "neutral"
        self.combat_available = combat_available  # Combat option available
        self.metadata = metadata or {}
        self.children: Dict[str, str] = {}  # Maps choice text to child node_id

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "narrative": self.narrative,
            "choices": self.choices,
            "is_ending": self.is_ending,
            "ending_type": self.ending_type,
            "combat_available": self.combat_available,
            "metadata": self.metadata,
            "children": self.children,
        }


class StoryTree:
    """Represents a complete story tree with nodes and transitions"""

    def __init__(self, campaign_id: str, nodes: Dict[str, StoryNode], root_node_id: str):
        self.campaign_id = campaign_id
        self.nodes = nodes
        self.root_node_id = root_node_id

    def get_node(self, node_id: str) -> Optional[StoryNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_root(self) -> StoryNode:
        """Get the root node"""
        return self.nodes[self.root_node_id]

    def find_node_by_keywords(self, keywords: List[str], current_node_id: Optional[str] = None) -> Optional[StoryNode]:
        """Find a node that matches given keywords in its narrative or choices"""
        search_nodes = (
            [self.nodes[current_node_id]]
            if current_node_id and current_node_id in self.nodes
            else list(self.nodes.values())
        )

        for node in search_nodes:
            narrative_lower = node.narrative.lower()
            for keyword in keywords:
                if keyword.lower() in narrative_lower:
                    return node
            for choice in node.choices:
                if any(keyword.lower() in choice.lower() for keyword in keywords):
                    return node
        return None

    def get_next_node_for_choice(self, current_node_id: str, choice_text: str) -> Optional[StoryNode]:
        """Get the next node based on a choice made"""
        current_node = self.nodes.get(current_node_id)
        if not current_node:
            return None

        # Find matching child node
        child_id = current_node.children.get(choice_text)
        if child_id:
            return self.nodes.get(child_id)

        # Fuzzy match if exact match fails
        choice_lower = choice_text.lower()
        for stored_choice, child_id in current_node.children.items():
            if choice_lower in stored_choice.lower() or stored_choice.lower() in choice_lower:
                return self.nodes.get(child_id)

        return None


class StoryTreeLoader:
    """Loads and manages story trees from JSON files"""

    STORY_TREES_DIR = Path(__file__).parent / "story_trees"

    @staticmethod
    def load_story_tree(campaign_id: str) -> Optional[StoryTree]:
        """Load a story tree from JSON file"""
        json_path = StoryTreeLoader.STORY_TREES_DIR / f"{campaign_id}.json"

        if not json_path.exists():
            logger.warning(f"Story tree file not found: {json_path}")
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            nodes = {}
            root_node_id = data.get("root_node_id")

            # Load all nodes
            for node_data in data.get("nodes", []):
                node = StoryNode(
                    node_id=node_data["node_id"],
                    narrative=node_data["narrative"],
                    choices=node_data.get("choices", []),
                    is_ending=node_data.get("is_ending", False),
                    ending_type=node_data.get("ending_type"),
                    combat_available=node_data.get("combat_available", False),
                    metadata=node_data.get("metadata", {}),
                )

                # Load children mappings
                for choice_text, child_id in node_data.get("children", {}).items():
                    node.children[choice_text] = child_id

                nodes[node.node_id] = node

            return StoryTree(campaign_id, nodes, root_node_id)

        except Exception as e:
            logger.error(f"Error loading story tree from {json_path}: {str(e)}")
            return None

    @staticmethod
    def get_available_story_trees() -> List[str]:
        """Get list of available story tree files"""
        if not StoryTreeLoader.STORY_TREES_DIR.exists():
            StoryTreeLoader.STORY_TREES_DIR.mkdir(parents=True, exist_ok=True)
            return []

        return [f.stem for f in StoryTreeLoader.STORY_TREES_DIR.glob("*.json")]

    @staticmethod
    def create_example_story_tree(campaign_id: str):
        """Create an example story tree JSON file"""
        example_tree = {
            "campaign_id": campaign_id,
            "root_node_id": "start_1",
            "nodes": [
                {
                    "node_id": "start_1",
                    "narrative": "You stand at the entrance of an ancient dungeon. Two paths lie before you.",
                    "choices": ["Take the left path", "Take the right path", "Investigate the entrance"],
                    "is_ending": False,
                    "children": {
                        "Take the left path": "left_path_1",
                        "Take the right path": "right_path_1",
                        "Investigate the entrance": "entrance_1",
                    },
                    "metadata": {"location": "Dungeon Entrance", "encounter_type": "decision"},
                },
                {
                    "node_id": "left_path_1",
                    "narrative": "You follow the left path and discover a treasure room! But it's guarded by a dragon.",
                    "choices": ["Fight the dragon", "Try to sneak past", "Retreat"],
                    "is_ending": False,
                    "children": {
                        "Fight the dragon": "ending_victory",
                        "Try to sneak past": "ending_neutral",
                        "Retreat": "start_1",
                    },
                    "metadata": {"location": "Treasure Room", "encounter_type": "combat"},
                },
                {
                    "node_id": "right_path_1",
                    "narrative": "The right path leads to a dead end with a riddle inscribed on the wall.",
                    "choices": ["Solve the riddle", "Turn back", "Search for hidden passages"],
                    "is_ending": False,
                    "children": {
                        "Solve the riddle": "ending_victory",
                        "Turn back": "start_1",
                        "Search for hidden passages": "ending_neutral",
                    },
                    "metadata": {"location": "Riddle Room", "encounter_type": "puzzle"},
                },
                {
                    "node_id": "entrance_1",
                    "narrative": "You find a hidden lever behind the entrance. Pulling it reveals a secret passage.",
                    "choices": ["Enter the secret passage", "Continue exploring the main paths"],
                    "is_ending": False,
                    "children": {
                        "Enter the secret passage": "ending_victory",
                        "Continue exploring the main paths": "start_1",
                    },
                    "metadata": {"location": "Secret Passage", "encounter_type": "discovery"},
                },
                {
                    "node_id": "ending_victory",
                    "narrative": "Congratulations! You have successfully completed the adventure and claimed the legendary treasure!",
                    "choices": [],
                    "is_ending": True,
                    "ending_type": "victory",
                    "metadata": {"location": "Treasure Vault", "ending_type": "victory"},
                },
                {
                    "node_id": "ending_neutral",
                    "narrative": "You escape the dungeon safely, but without the treasure. Perhaps another time...",
                    "choices": [],
                    "is_ending": True,
                    "ending_type": "neutral",
                    "metadata": {"location": "Dungeon Exit", "ending_type": "neutral"},
                },
            ],
        }

        # Create directory if it doesn't exist
        StoryTreeLoader.STORY_TREES_DIR.mkdir(parents=True, exist_ok=True)

        # Write example file
        json_path = StoryTreeLoader.STORY_TREES_DIR / f"{campaign_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(example_tree, f, indent=2, ensure_ascii=False)

        return json_path


# Import logger
import logging

logger = logging.getLogger(__name__)
