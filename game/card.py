from dataclasses import dataclass


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
        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_values[self.rank]

