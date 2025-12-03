import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append("../")

from game.game import BridgePlay
from game.game_state import PlayerType
from agents.random_agent import RandomAgent
from agents.high_card_agent import HighCardAgent
from agents.low_card_agent import LowCardAgent
from agents.rule_based_agent import RuleBasedAgent
from typing import Dict, List, Tuple
import time


class GameRunner:
    
    def __init__(
        self,
        defender1_agent_class,
        dummy_agent_class,
        defender2_agent_class,
        lead_agent_class,
        contract: int = 7
    ):
        """
        Initialize the game runner.
        
        Args:
            defender1_agent_class: Class for DEFENDER_1 agent
            dummy_agent_class: Class for DUMMY agent
            defender2_agent_class: Class for DEFENDER_2 agent  
            lead_agent_class: Class for LEAD agent
            contract: Number of tricks to bid (default: 7)
        """
        self.defender1_class = defender1_agent_class
        self.dummy_class = dummy_agent_class
        self.defender2_class = defender2_agent_class
        self.lead_class = lead_agent_class
        self.contract = contract
        
        # Statistics
        self.results: List[Dict] = []
        
    def run_game(self) -> Dict:
        """
        Run a single game and return the result.
        
        Returns:
            Dictionary with game statistics
        """
        # Create fresh agents for this game
        defender1 = self.defender1_class(PlayerType.DEFENDER_1)
        dummy = self.dummy_class(PlayerType.DUMMY)
        defender2 = self.defender2_class(PlayerType.DEFENDER_2)
        lead = self.lead_class(PlayerType.LEAD)
        
        # Create and play game
        game = BridgePlay(
            contract=self.contract,
            defender1_agent=defender1,
            dummy_agent=dummy,
            defender2_agent=defender2,
            lead_agent=lead
        )
        
        result = game.play_game()
        
        # Extract statistics
        stats = {
            'lead_tricks': result.lead_tricks,
            'defender_tricks': result.defender_tricks,
            'lead_score': result.lead_score,
            'defender_score': result.defender_score,
            'lead_won': result.lead_score > 0,
            'lead_made_contract': result.lead_tricks >= self.contract,
            'contract': self.contract,
        }
        
        return stats
    
    def run_games(self, n_games: int, verbose: bool = True) -> Dict:
        """
        Run multiple games and collect statistics.
        
        Args:
            n_games: Number of games to run
            verbose: Whether to print progress
            
        Returns:
            Dictionary with aggregated statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running {n_games} games")
            print(f"  Defender 1: {self.defender1_class.__name__}")
            print(f"  Dummy: {self.dummy_class.__name__}")
            print(f"  Defender 2: {self.defender2_class.__name__}")
            print(f"  Lead: {self.lead_class.__name__}")
            print(f"  Contract: {self.contract}")
            print(f"{'='*60}\n")
        
        start_time = time.time()
        self.results = []
        
        for i in range(n_games):
            result = self.run_game()
            self.results.append(result)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_games} games...")
        
        elapsed = time.time() - start_time
        
        # Compute aggregate statistics
        stats = self._compute_statistics()
        stats['n_games'] = n_games
        stats['elapsed_time'] = elapsed
        stats['games_per_second'] = n_games / elapsed if elapsed > 0 else 0
        
        if verbose:
            self._print_statistics(stats)
        
        return stats
    
    def _compute_statistics(self) -> Dict:
        """Compute aggregate statistics from all game results."""
        if not self.results:
            return {}
        
        lead_wins = sum(1 for r in self.results if r['lead_won'])
        lead_made = sum(1 for r in self.results if r['lead_made_contract'])
        
        total_lead_score = sum(r['lead_score'] for r in self.results)
        total_lead_tricks = sum(r['lead_tricks'] for r in self.results)
        total_defender_tricks = sum(r['defender_tricks'] for r in self.results)
        
        n = len(self.results)
        
        return {
            'lead_win_rate': lead_wins / n,
            'lead_contract_rate': lead_made / n,
            'avg_lead_score': total_lead_score / n,
            'avg_lead_tricks': total_lead_tricks / n,
            'avg_defender_tricks': total_defender_tricks / n,
        }
    
    def _print_statistics(self, stats: Dict):
        """Print formatted statistics."""
        print(f"\n{'='*60}")
        print("Results:")
        print(f"{'='*60}")
        print(f"  Games played: {stats['n_games']}")
        print(f"  Time elapsed: {stats['elapsed_time']:.2f}s")
        print(f"  Games/second: {stats['games_per_second']:.1f}")
        print()
        print(f"  Lead Team Performance:")
        print(f"    Win rate (beat contract): {stats['lead_win_rate']*100:.1f}%")
        print(f"    Made contract rate: {stats['lead_contract_rate']*100:.1f}%")
        print(f"    Average score: {stats['avg_lead_score']:.2f}")
        print(f"    Average tricks: {stats['avg_lead_tricks']:.2f}/13")
        print()
        print(f"  Defender Team Performance:")
        print(f"    Win rate: {(1-stats['lead_win_rate'])*100:.1f}%")
        print(f"    Average tricks: {stats['avg_defender_tricks']:.2f}/13")
        print(f"{'='*60}\n")


def run_baseline_comparison():
    """
    Run a comparison of baseline agents.
    
    This demonstrates how to compare different agent strategies.
    """
    print("\n" + "="*60)
    print("BASELINE AGENT COMPARISON")
    print("="*60)
    
    # Define different agent configurations to test
    configs = [
        {
            'name': 'All Random',
            'defender1': RandomAgent,
            'dummy': RandomAgent,
            'defender2': RandomAgent,
            'lead': RandomAgent,
        },
        {
            'name': 'High Lead vs Random Defenders',
            'defender1': RandomAgent,
            'dummy': RandomAgent,
            'defender2': RandomAgent,
            'lead': HighCardAgent,
        },
        {
            'name': 'Low Lead vs Random Defenders',
            'defender1': RandomAgent,
            'dummy': RandomAgent,
            'defender2': RandomAgent,
            'lead': LowCardAgent,
        },
        {
            'name': 'Random Lead vs High Defenders',
            'defender1': HighCardAgent,
            'dummy': RandomAgent,
            'defender2': HighCardAgent,
            'lead': RandomAgent,
        },
        {
            'name': 'High Everyone',
            'defender1': HighCardAgent,
            'dummy': HighCardAgent,
            'defender2': HighCardAgent,
            'lead': HighCardAgent,
        },
        {
            'name': 'Rule-Based Everyone',
            'defender1': RuleBasedAgent,
            'dummy': RuleBasedAgent,
            'defender2': RuleBasedAgent,
            'lead': RuleBasedAgent,
        },
        {
            'name': 'Rule-Based Lead vs Random',
            'defender1': RandomAgent,
            'dummy': RuleBasedAgent,
            'defender2': RandomAgent,
            'lead': RuleBasedAgent,
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        runner = GameRunner(
            defender1_agent_class=config['defender1'],
            dummy_agent_class=config['dummy'],
            defender2_agent_class=config['defender2'],
            lead_agent_class=config['lead'],
            contract=7
        )
        
        stats = runner.run_games(n_games=500, verbose=False)
        stats['config_name'] = config['name']
        results.append(stats)
        
        print(f"  Lead win rate: {stats['lead_win_rate']*100:.1f}%")
        print(f"  Avg lead score: {stats['avg_lead_score']:.2f}")
    
    # Print summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Configuration':<40} {'Win Rate':<12} {'Avg Score':<12}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['config_name']:<40} {r['lead_win_rate']*100:>6.1f}%      {r['avg_lead_score']:>8.2f}")
    print(f"{'='*60}\n")


def main():
    """Main entry point for the starter game."""
    print("\n" + "="*60)
    print("Bridge Play RL - Starter Game")
    print("="*60)
    
    # Example 1: Run a single configuration
    print("\nExample 1: Single Configuration")
    runner = GameRunner(
        defender1_agent_class=RandomAgent,
        dummy_agent_class=RandomAgent,
        defender2_agent_class=RandomAgent,
        lead_agent_class=HighCardAgent,
        contract=7
    )
    runner.run_games(n_games=1000)
    
    # Example 2: Compare different strategies
    print("\n" + "="*60)
    print("Example 2: Strategy Comparison")
    print("="*60)
    run_baseline_comparison()

if __name__ == "__main__":
    main()

