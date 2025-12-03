"""
Rule-Based Agent for Bridge Play (Fixed Order)

Play order is FIXED: Defender1 → Dummy → Defender2 → Lead
This never changes - D1 leads every trick.

Strategies are adapted for this fixed order game.
"""

from game.agent import BridgePlayAgent
from game.game_state import PlayObservation, PlayerType
from game.card import Card
from typing import List, Optional


class RuleBasedAgent(BridgePlayAgent):
    """
    Agent using rule-based strategy for fixed-order Bridge Play.
    
    Fixed order: D1 → Dummy → D2 → Lead (every trick)
    """
    
    def __init__(self, player_type: PlayerType):
        super().__init__(player_type)
    
    def get_action(self, observation: PlayObservation) -> Card:
        """Dispatch to role-specific strategy function."""
        if self.player_type == PlayerType.DEFENDER_1:
            return self.get_action_defender1(observation)
        elif self.player_type == PlayerType.DUMMY:
            return self.get_action_dummy(observation)
        elif self.player_type == PlayerType.DEFENDER_2:
            return self.get_action_defender2(observation)
        else:
            return self.get_action_lead(observation)
    
    # ==================== ROLE-BASED STRATEGY FUNCTIONS ====================
    
    def get_action_defender1(self, observation: PlayObservation) -> Card:
        """
        Defender 1 - ALWAYS leads (position 0).
        
        Advantages:
        - Can see dummy's hand before deciding what to lead
        - Sets the suit for the trick
        
        Strategy:
        - Lead suits where dummy is WEAK (few cards, low ranks)
        - Lead high cards (A, K) to win immediately
        - Avoid leading suits where dummy has strength
        """
        hand = observation.legal_actions
        dummy_hand = observation.dummy_hand
        
        # Analyze dummy's weakness by suit
        dummy_suits = self._group_by_suit(dummy_hand) if dummy_hand else {}
        my_suits = self._group_by_suit(hand)
        
        # strat 1: Lead Aces to win immediately
        aces = [c for c in hand if c.rank == 'A']
        if aces:
            # Prefer ace in suit where dummy is weak
            for ace in aces:
                dummy_count = len(dummy_suits.get(ace.suit, []))
                if dummy_count <= 2:  # Dummy weak in this suit
                    return ace
            return aces[0]  # Any ace
        
        # strat 2: Lead Kings if we also have the Queen (safe lead)
        for suit, cards in my_suits.items():
            ranks = {c.rank for c in cards}
            if 'K' in ranks and 'Q' in ranks:
                return next(c for c in cards if c.rank == 'K')
        
        # strat 3: Lead from suit where dummy is weakest
        best_suit = None
        best_score = -1
        
        for suit, my_cards in my_suits.items():
            dummy_cards = dummy_suits.get(suit, [])
            dummy_strength = sum(c.rank_value() for c in dummy_cards)
            dummy_count = len(dummy_cards)
            
            # Score: prefer suits where dummy is weak (low count, low strength)
            # Higher score = better suit to lead
            weakness_score = (4 - dummy_count) * 10 + (56 - dummy_strength)
            
            if weakness_score > best_score:
                best_score = weakness_score
                best_suit = suit
        
        if best_suit and best_suit in my_suits:
            # Lead highest card from that suit
            return max(my_suits[best_suit], key=lambda c: c.rank_value())
        
        # Fallback: lead highest card overall
        return max(hand, key=lambda c: c.rank_value())
    
    def get_action_dummy(self, observation: PlayObservation) -> Card:
        """
        Dummy - ALWAYS plays second (position 1).
        
        PERFECT INFORMATION: Sees own hand AND Lead's hand (via dummy_hand).
        
        Order: D1 played → Dummy → D2 → Lead
        
        Strategy:
        - Check if Lead can beat D1's card in the led suit
        - If Lead CAN win: play LOW (let partner handle it economically)
        - If Lead CANNOT win: Dummy should try to win now
        - Coordinate to ensure team wins with minimum resources
        """
        hand = observation.legal_actions
        lead_hand = observation.dummy_hand  # For Dummy, this shows Lead's cards (opposite only in this case)
        d1_card = observation.current_trick[0]
        led_suit = d1_card.suit
        
        # Cards we can play in the led suit
        my_cards_in_suit = [c for c in hand if c.suit == led_suit]
        
        # Cards Lead has in the led suit
        lead_cards_in_suit = [c for c in lead_hand if c.suit == led_suit] if lead_hand else []
        
        if my_cards_in_suit:
            # Check if Lead can beat D1's card
            lead_can_beat = any(c.rank_value() > d1_card.rank_value() for c in lead_cards_in_suit)
            
            # My cards that can beat D1
            my_beating_cards = [c for c in my_cards_in_suit if c.rank_value() > d1_card.rank_value()]
            
            if lead_can_beat:
                # Lead can win - play lowest to conserve our high cards
                return min(my_cards_in_suit, key=lambda c: c.rank_value())
            
            elif my_beating_cards:
                # Lead can't win but we can - beat with minimum needed
                # (D2 plays next, so we need to beat potential D2 cards too)
                # Play our highest beater to maximize chance of holding through D2
                return max(my_beating_cards, key=lambda c: c.rank_value())
            
            else:
                # Neither can beat - play lowest
                return min(my_cards_in_suit, key=lambda c: c.rank_value())
        
        # Can't follow suit - discard lowest from weakest suit
        # Consider Lead's hand: discard from suit where Lead is strong
        if lead_hand:
            lead_suits = self._group_by_suit(lead_hand)
            # Find suit where Lead is strongest (we can discard from there)
            for suit in ['S', 'H', 'D', 'C']:  # Check in order
                if suit in lead_suits and len(lead_suits[suit]) >= 3:
                    discards = [c for c in hand if c.suit == suit]
                    if discards:
                        return min(discards, key=lambda c: c.rank_value())
        
        # Fallback: discard lowest overall
        return min(hand, key=lambda c: c.rank_value())
    
    def get_action_defender2(self, observation: PlayObservation) -> Card:
        """
        Defender 2 - ALWAYS plays third (position 2).
        
        Sees: D1's card and Dummy's card
        Partner: D1 (played first)
        
        Strategy:
        - If D1 (partner) is winning: play LOW to conserve
        - If Dummy is winning: try to beat it to force Lead's hand
        - Force Lead to use high cards
        """
        hand = observation.legal_actions
        trick = observation.current_trick  # [D1, Dummy]
        led_suit = trick[0].suit
        
        cards_in_suit = [c for c in hand if c.suit == led_suit]
        
        if cards_in_suit:
            winning_card = max(trick, key=lambda c: c.card_value(led_suit))
            d1_winning = (trick[0] == winning_card)
            
            if d1_winning:
                # Partner winning - play lowest to conserve
                return min(cards_in_suit, key=lambda c: c.rank_value())
            
            # Dummy is winning - try to beat it
            beating = [c for c in cards_in_suit 
                      if c.rank_value() > winning_card.rank_value()]
            if beating:
                # Beat with minimum card needed
                return min(beating, key=lambda c: c.rank_value())
            
            # Can't beat - play lowest
            return min(cards_in_suit, key=lambda c: c.rank_value())
        
        # Can't follow suit - discard lowest
        return min(hand, key=lambda c: c.rank_value())
    
    def get_action_lead(self, observation: PlayObservation) -> Card:
        """
        Lead - ALWAYS plays fourth/last (position 3).
        
        perfect information: Sees all 3 cards before deciding.
        Partner: Dummy (played second)
        
        Strategy:
        - If Dummy (partner) is winning: play LOWEST (save resources)
        - If defender winning: beat with MINIMUM card needed
        - If can't win: dump lowest card
        """
        hand = observation.legal_actions
        trick = observation.current_trick  # [D1, Dummy, D2]
        led_suit = trick[0].suit
        
        cards_in_suit = [c for c in hand if c.suit == led_suit]
        
        if cards_in_suit:
            winning_card = max(trick, key=lambda c: c.card_value(led_suit))
            dummy_winning = (trick[1] == winning_card)
            
            if dummy_winning:
                # Partner winning - play lowest (economical)
                return min(cards_in_suit, key=lambda c: c.rank_value())
            
            # Opponent winning - beat with minimum
            beating = [c for c in cards_in_suit 
                      if c.rank_value() > winning_card.rank_value()]
            if beating:
                return min(beating, key=lambda c: c.rank_value())
            
            # Can't win - dump lowest
            return min(cards_in_suit, key=lambda c: c.rank_value())
        
        # Can't follow suit - discard lowest
        return min(hand, key=lambda c: c.rank_value())
    
    
    def _group_by_suit(self, cards: List[Card]) -> dict:
        """Group cards by suit."""
        suits = {}
        for card in cards:
            if card.suit not in suits:
                suits[card.suit] = []
            suits[card.suit].append(card)
        return suits
