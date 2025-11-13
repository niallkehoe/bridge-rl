from game.agent import BridgePlayAgent
from game.game_state import PlayObservation, PlayerType
from game.card import Card


class LowCardAgent(BridgePlayAgent):
    """
    Agent that always plays the lowest legal card.
    
    This agent tries to conserve high cards for later tricks.
    """
    
    def __init__(self, player_type: PlayerType):
        super().__init__(player_type)
    
    def get_action(self, observation: PlayObservation) -> Card:
        """
        Select the lowest value card from legal actions.
        
        Args:
            observation: Current game state observation
            
        Returns:
            Lowest legal card
        """
        return min(observation.legal_actions, key=lambda card: card.rank_value())

