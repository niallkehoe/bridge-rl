from dataclasses import dataclass


rank_order = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

@dataclass
class Card:
    """Represents a playing card."""
    suit: str  # 'C', 'D', 'H', 'S' (Clubs, Diamonds, Hearts, Spades)
    rank: str  # '2'-'9', 'T', 'J', 'Q', 'K', 'A'
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank
    
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))
    
    
    def rank_value(self) -> int:
        """Returns rank value (2=2, ..., A=14) for comparing cards in same suit."""
        return rank_order[self.rank]
    
    def card_value(self, led_suit: str) -> int:
        """
        Returns card value relative to led suit for trick comparison.
        
        Args:
            led_suit: The suit that was led ('C', 'D', 'H', or 'S')
            
        Returns:
            1-13 if card matches led suit (2=1, 3=2, ..., A=13)
            0 if card doesn't match led suit (can't win the trick)
            
        Usage:
            winner = max(trick, key=lambda c: c.card_value(led_suit))
        """
        if self.suit != led_suit:
            return 0
        
        return rank_order[self.rank]

