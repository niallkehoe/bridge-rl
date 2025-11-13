import random
from game.agent import BridgePlayAgent
from game.game_state import PlayObservation, PlayerType
from game.card import Card


class RandomAgent(BridgePlayAgent):
    """
    Agent that plays a random legal card.
    
    This is the simplest possible agent and serves as a baseline.
    """
    
    def __init__(self, player_type: PlayerType):
        super().__init__(player_type)
    
    def get_action(self, observation: PlayObservation) -> Card:
        """
        Select a random card from legal actions.
        
        Args:
            observation: Current game state observation
            
        Returns:
            Random legal card
        """
        return random.choice(observation.legal_actions)

