"""
game_state.py

Game state tree structure for managing D&D game sessions.
Tracks state transitions, agent routing, and game history.
"""

from enum import Enum
from typing import Optional, Dict, List, Any
from datetime import datetime
from uuid import uuid4


class GameStateType(Enum):
    """Types of game states"""
    NARRATION = "narration"      # Story/exploration mode
    COMBAT = "combat"            # Combat encounter
    DIALOGUE = "dialogue"        # NPC conversation
    DECISION = "decision"        # Player choice point
    GAME_OVER = "game_over"      # End state


class AgentType(Enum):
    """Types of agents handling states"""
    NARRATOR = "narrator"        # Finetuned narrator model
    COMBAT = "combat"            # Combat agent
    ORCHESTRATOR = "orchestrator" # Decision router


class GameStateNode:
    """Represents a single state in the game's state tree"""

    def __init__(
        self,
        state_type: GameStateType,
        agent: AgentType,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.state_type = state_type
        self.agent = agent
        self.parent_id = parent_id
        self.children: List[str] = []  # Child node IDs
        self.created_at = datetime.now()

        # State-specific data
        self.metadata = metadata or {}
        self.narrative_text: Optional[str] = None
        self.combat_session_id: Optional[str] = None
        self.player_action: Optional[str] = None
        self.agent_response: Optional[str] = None

        # Transition detection
        self.transition_triggered = False
        self.next_state_type: Optional[GameStateType] = None

        # Rule validation tracking (NEW)
        self.rule_validation: Optional[Dict] = None
        self.applicable_rules: Optional[str] = None
        self.was_validated: bool = False
        self.validation_errors: List[str] = []

    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization"""
        return {
            "id": self.id,
            "state_type": self.state_type.value,
            "agent": self.agent.value,
            "parent_id": self.parent_id,
            "children": self.children,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "narrative_text": self.narrative_text,
            "combat_session_id": self.combat_session_id,
            "player_action": self.player_action,
            "agent_response": self.agent_response,
            "transition_triggered": self.transition_triggered,
            "next_state_type": self.next_state_type.value if self.next_state_type else None,
            "rule_validation": self.rule_validation,
            "applicable_rules": self.applicable_rules,
            "was_validated": self.was_validated,
            "validation_errors": self.validation_errors
        }


class GameStateTree:
    """Manages the entire game state tree"""

    def __init__(self):
        self.nodes: Dict[str, GameStateNode] = {}
        self.root_id: Optional[str] = None
        self.current_node_id: Optional[str] = None

    def create_root(self, state_type: GameStateType = GameStateType.NARRATION) -> GameStateNode:
        """Initialize the game with a root narration node"""
        root = GameStateNode(
            state_type=state_type,
            agent=AgentType.NARRATOR,
            metadata={"is_root": True}
        )
        self.nodes[root.id] = root
        self.root_id = root.id
        self.current_node_id = root.id
        return root

    def add_child(
        self,
        parent_id: str,
        state_type: GameStateType,
        agent: AgentType,
        metadata: Optional[Dict] = None
    ) -> GameStateNode:
        """Add a new state as a child of the current state"""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")

        child = GameStateNode(
            state_type=state_type,
            agent=agent,
            parent_id=parent_id,
            metadata=metadata
        )

        self.nodes[child.id] = child
        self.nodes[parent_id].children.append(child.id)
        return child

    def transition_to(self, node_id: str):
        """Move to a different node in the tree"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        self.current_node_id = node_id

    def get_current(self) -> Optional[GameStateNode]:
        """Get the current active node"""
        if self.current_node_id:
            return self.nodes.get(self.current_node_id)
        return None

    def get_node(self, node_id: str) -> Optional[GameStateNode]:
        """Get a specific node by ID"""
        return self.nodes.get(node_id)

    def get_path_from_root(self) -> List[GameStateNode]:
        """Get the path from root to current node"""
        if not self.current_node_id:
            return []

        path = []
        node = self.nodes[self.current_node_id]
        while node:
            path.insert(0, node)
            node = self.nodes.get(node.parent_id) if node.parent_id else None
        return path

    def to_dict(self) -> Dict:
        """Convert entire tree to dictionary"""
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_id": self.root_id,
            "current_node_id": self.current_node_id
        }
