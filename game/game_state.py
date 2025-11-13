from dataclasses import dataclass, field
from typing import List, Optional
from enum import IntEnum
from game.card import Card


class PlayerType(IntEnum):
    """Player type enumeration."""
    DEFENDER_1 = 0
    DUMMY = 1
    DEFENDER_2 = 2
    LEAD = 3


@dataclass
class PlayObservation:
    """
    Observation provided to an agent during the play phase.
    
    Attributes:
        hand: List of cards in the agent's hand (updated as cards are played)
        current_trick: List of cards played in current trick (position indicates player)
        trick_index: Current trick number (0-12)
        contract: Number of tricks the lead team bid to win
        legal_actions: List of legal cards that can be played from this hand
        player_id: The ID of the player making this decision (0, 1, 2, or 3)
        dummy_hand: Dummy's hand (visible to all players after opening lead, None before)
    """
    hand: List[Card]
    current_trick: List[Card]
    trick_index: int
    contract: int
    legal_actions: List[Card]
    player_id: int
    dummy_hand: List[Card] = None


@dataclass
class GameResult:
    """Result of a completed Bridge Play game (stge 2)."""
    lead_tricks: int
    defender_tricks: int
    contract: int
    lead_score: int
    defender_score: int
    trick_history: List[List[Card]] = field(default_factory=list)
