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
    runner = GameRunner(
        defender1_agent_class=DeepQLearningAgent,
        dummy_agent_class=DeepQLearningAgent,
        defender2_agent_class=DeepQLearningAgent,
        lead_agent_class=DeepQLearningAgent,
        contract=7
    )
    runner.run_games(n_games=1000)

if __name__ == "__main__":
    main()