from game.agent import BridgePlayAgent
from game.game_state import PlayObservation, PlayerType
from game.card import Card


class HighCardAgent(BridgePlayAgent):
    """
    Agent that always plays the highest legal card.
    
    This is a simple heuristic agent that tries to win tricks by playing high cards.
    """
    
    def __init__(self, player_type: PlayerType):
        super().__init__(player_type)
    
    def get_action(self, observation: PlayObservation) -> Card:
        """
        Select the highest value card from legal actions.
        
        Args:
            observation: Current game state observation
            
        Returns:
            Highest legal card
        """
        return max(observation.legal_actions, key=lambda card: card.rank_value())

