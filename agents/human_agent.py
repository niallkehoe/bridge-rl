from game.agent import BridgePlayAgent
from game.game_state import PlayObservation, PlayerType
from game.card import Card


class HumanAgent(BridgePlayAgent):
    """
    Agent that prompts for human input via console.
    """
    
    def __init__(self, player_type: PlayerType):
        super().__init__(player_type)
    
    def get_action(self, observation: PlayObservation) -> Card:
        """
        Prompt human player to select a card.
        
        Args:
            observation: Current game state observation
            
        Returns:
            Card selected by human player
        """
        # Print deatil of game state
        print(f"\n=== Player {observation.player_id}'s Turn ===")
        print(f"Trick {observation.tricks_played + 1}/13")
        print(f"Contract: {observation.contract} tricks")
        print(f"\nCurrent trick: {observation.current_trick}")
        print(f"\nYour hand: {sorted(observation.hand, key=lambda c: (c.suit, c.rank_value()))}")
        
        if observation.dummy_hand:
            print(f"Dummy's hand: {sorted(observation.dummy_hand, key=lambda c: (c.suit, c.rank_value()))}")
        
        print(f"\nLegal actions:")
        for i, card in enumerate(observation.legal_actions):
            print(f"  {i}: {card}")
        
        # prompt for the index of the action
        while True:
            try:
                choice = int(input("\nEnter card number to play: "))
                if 0 <= choice < len(observation.legal_actions):
                    return observation.legal_actions[choice]
                else:
                    print("Invalid choice. Try again.")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Try again.")

