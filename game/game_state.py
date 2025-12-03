from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
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
        tricks_played: Number of tricks completed so far (0-12)
        tricks_won: Number of tricks won by this player's team so far
        contract: Number of tricks the lead team bid to win
        legal_actions: List of legal cards that can be played from this hand
        player_id: The ID of the player making this decision (0, 1, 2, or 3)
        dummy_hand: Partner's visible hand (Dummy sees Lead's hand, others see Dummy's hand)
            In the dummy's perspective, the lead's hand is the alternative hand [cannot use these cards]
            but should be aware of there values.
    """
    hand: List[Card]
    current_trick: List[Card]
    tricks_played: int
    tricks_won: int
    contract: int
    legal_actions: List[Card]
    player_id: int
    dummy_hand: List[Card] = None


@dataclass
class GameResult:
    """Result of a completed Bridge Play game (stage 2)."""
    lead_tricks: int
    defender_tricks: int
    contract: int
    lead_score: int
    defender_score: int
    trick_history: List[List[Card]] = field(default_factory=list)
    # Per-player history: PlayerType -> List of (observation, action) tuples
    observation_action_history: Dict['PlayerType', List[Tuple['PlayObservation', Card]]] = field(default_factory=dict)
