"""
Bridge Play Game Engine (stage 2)
"""

import random
from typing import List, Tuple, Dict
from game.card import Card
from game.game_state import PlayObservation, GameResult, PlayerType
from game.agent import BridgePlayAgent


class BridgePlay:
    """
    Simplified Bridge Play game (no bidding phase).
    
    Players:
        0: Defender 1 (DEFENDER_1)
        1: Dummy (DUMMY)
        2: Defender 2 (DEFENDER_2)
        3: Lead (LEAD)
        
    Teams:
        Lead Team: Players 1 (Dummy) and 3 (Lead)
        Defenders: Players 0 and 2
        
    Scoring:
        Lead team: (tricks_won - contract) * 20
        Defender team: -(tricks_won - contract) * 20
    """
    
    # Card constants
    SUITS = ['C', 'D', 'H', 'S']  # Clubs, Diamonds, Hearts, Spades
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    def __init__(self, 
                 contract: int,
                 defender1_agent: BridgePlayAgent,
                 dummy_agent: BridgePlayAgent,
                 defender2_agent: BridgePlayAgent,
                 lead_agent: BridgePlayAgent
    ):
        """
        Initialize a Bridge Play game.
        
        Args:
            contract: Number of tricks the lead team bid to win
            defender1_agent: Agent for player 0 (Defender 1)
            dummy_agent: Agent for player 1 (Dummy)
            defender2_agent: Agent for player 2 (Defender 2)
            lead_agent: Agent for player 3 (Lead)
        """
        self.contract = contract
        self.agents = {
            PlayerType.DEFENDER_1: defender1_agent,
            PlayerType.DUMMY: dummy_agent,
            PlayerType.DEFENDER_2: defender2_agent,
            PlayerType.LEAD: lead_agent
        }
        
        # Game state
        self.hands: Dict[int, List[Card]] = {0: [], 1: [], 2: [], 3: []}
        self.current_trick: List[Card] = []
        self.trick_history: List[List[Card]] = []

        # trick leader initialized on play
        self.tricks_won = {0: 0, 1: 0, 2: 0, 3: 0}  # Per player
        self.current_player = PlayerType.DEFENDER_1  # Defender 1 leads first trick
        self.trick_index = 0
        
    def create_deck(self) -> List[Card]:
        """Create a standard 52-card deck."""
        return [Card(suit, rank) for suit in self.SUITS for rank in self.RANKS]
    
    def deal(self):
        """Deal cards to all players (13 each)."""
        deck = self.create_deck()
        random.shuffle(deck)
        
        for i in range(4):
            self.hands[i] = deck[i*13:(i+1)*13]
    
    def get_legal_actions(self, player_id: int) -> List[Card]:
        """
        Get legal cards for a player to play.
        
        Rules:
        - If leading the trick (first to play), any card is legal
        - Otherwise, must follow suit if possible
        - If cannot follow suit, any card is legal
        """
        hand = self.hands[player_id]
        
        if not self.current_trick:
            # Leading the trick - any card is legal
            return hand.copy()
        
        # Must follow suit if possible
        led_suit = self.current_trick[0].suit
        cards_in_suit = [card for card in hand if card.suit == led_suit]
        
        if cards_in_suit:
            return cards_in_suit
        else:
            # Cannot follow suit - any card is legal
            return hand.copy()
    
    def determine_trick_winner(self) -> int:
        """
        Determine the winner of the current trick.
        
        Returns:
            Player ID of the trick winner
        """
        if not self.current_trick:
            raise ValueError("No cards in current trick")
        
        led_suit = self.current_trick[0].suit
        
        # Find the highest card in the led suit
        highest_value = -1
        winner_position = -1
        
        for position, card in enumerate(self.current_trick):
            if card.suit == led_suit:
                if card.rank_value() > highest_value:
                    highest_value = card.rank_value()
                    winner_position = position
        
        # Convert position to player ID
        winner = winner_position % 4
        return winner
    
    def play_card(self, player_id: int, card: Card):
        """
        Play a card from a player's hand.
        
        Args:
            player_id: ID of the player playing the card
            card: Card being played
        """
        if card not in self.hands[player_id]:
            raise ValueError(f"Card {card} not in player {player_id}'s hand")
        
        legal_actions = self.get_legal_actions(player_id)
        if card not in legal_actions:
            raise ValueError(f"Card {card} is not a legal action for player {player_id}")
        
        # Remove card from hand and add to current trick
        self.hands[player_id].remove(card)
        self.current_trick.append(card)
    
    def get_observation(self, player_id: int) -> PlayObservation:
        """
        Create an observation for a player.
        
        Args:
            player_id: ID of the player (0, 1, 2, or 3)
            
        Returns:
            PlayObservation object
        """
        # Dummy hand is visible to all players always (slight hack, should be after tick 0 and play 0)
        dummy_hand = self.hands[PlayerType.DUMMY].copy()
        
        return PlayObservation(
            hand=self.hands[player_id].copy(),
            current_trick=self.current_trick.copy(),
            trick_index=self.trick_index,
            contract=self.contract,
            legal_actions=self.get_legal_actions(player_id),
            player_id=player_id,
            dummy_hand=dummy_hand
        )
    
    def play_trick(self):
        """Play a complete trick (4 cards)."""
        self.current_trick = []
        
        for _ in range(4):
            # Get the agent for current player
            agent = self.agents[self.current_player]
            
            # Get observation and action
            observation = self.get_observation(self.current_player)
            card = agent.get_action(observation)
            
            # Play the card
            self.play_card(self.current_player, card)
            
            # Move to next player
            self.current_player = (self.current_player + 1) % 4
        
        # Determine winner
        winner = self.determine_trick_winner()
        self.tricks_won[winner] += 1
        
        # Save trick to history
        self.trick_history.append(self.current_trick.copy())
        
        # Winner leads next trick
        self.current_player = winner
        self.trick_index += 1
    
    def calculate_scores(self) -> Tuple[int, int]:
        """
        Calculate final scores for both teams.
        
        Returns:
            Tuple of (lead_score, defender_score)
        """
        lead_tricks = self.tricks_won[PlayerType.DUMMY] + self.tricks_won[PlayerType.LEAD]
        lead_score = (lead_tricks - self.contract) * 20
        defender_score = -lead_score
        
        return lead_score, defender_score
    
    def play_game(self) -> GameResult:
        """
        Play a complete game of Bridge Play.
        
        Returns:
            GameResult with final scores and game history
        """
        # Deal cards
        self.deal()
        
        # Play all 13 tricks
        for _ in range(13):
            self.play_trick()
        
        # Calculate scores
        lead_score, defender_score = self.calculate_scores()
        lead_tricks = self.tricks_won[PlayerType.DUMMY] + self.tricks_won[PlayerType.LEAD]
        defender_tricks = self.tricks_won[PlayerType.DEFENDER_1] + self.tricks_won[PlayerType.DEFENDER_2]
        
        return GameResult(
            lead_tricks=lead_tricks,
            defender_tricks=defender_tricks,
            contract=self.contract,
            lead_score=lead_score,
            defender_score=defender_score,
            trick_history=self.trick_history,
        )
