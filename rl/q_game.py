import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append("../")

from game.game import BridgePlay
from game.game_state import PlayerType, PlayObservation, GameResult
from game.card import Card
from agents.random_agent import RandomAgent
from agents.high_card_agent import HighCardAgent
from agents.low_card_agent import LowCardAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.q_agent import DeepQLearningAgent
from typing import Dict, List, Tuple, Callable, Optional
import time
from starter_game import GameRunner, run_baseline_comparison


def main():
    """Main entry point for the Q game."""
    print("\n" + "="*60)
    print("Bridge Play RL - Q Game")
    print("="*60)
    
    # Example 1: Run a single configuration
    print("\nExample 1: Single Configuration")

    defender1 = RandomAgent(PlayerType.DEFENDER_1)
    dummy = RandomAgent(PlayerType.DUMMY)
    defender2 = RandomAgent(PlayerType.DEFENDER_2)
    lead = DeepQLearningAgent(PlayerType.LEAD)

    def on_game_end(observation_action_history, lead_score, defender_score):
        #dummy.on_game_end(observation_action_history, lead_score, defender_score)
        lead.on_game_end(observation_action_history, lead_score, defender_score)

    runner = GameRunner(
        defender1_agent=defender1,
        dummy_agent=dummy,
        defender2_agent=defender2,
        lead_agent=lead,
        contract=6,
        on_game_end = on_game_end
    )
    runner.run_games(n_games=100000)

if __name__ == "__main__":
    main()