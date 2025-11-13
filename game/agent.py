from abc import ABC, abstractmethod
from game.card import Card
from game.game_state import PlayObservation, PlayerType


class BridgePlayAgent(ABC):
    """
    Abstract base class for Bridge Play (stage 2) agents.
    
    All agents must implement the get_action method which receives an Observation
    and returns a Card to play.
    
    Agents are initialized with their PlayerTpe (DEFENDER_1, DUMMY, DEFENDER_2, or LEAD).
    """
    
    def __init__(self, player_type: PlayerType):
        self.player_type = player_type
    
    @abstractmethod
    def get_action(self, observation: PlayObservation) -> Card:
        """
        Select a card to play given the current observation.
        
        Args:
            observation: Current game state observation
            
        Returns:
            Card to play (must be in observation.legal_actions)
        """
        pass
