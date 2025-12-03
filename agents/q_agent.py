from game.agent import BridgePlayAgent
from game.game_state import PlayObservation, PlayerType
from game.card import Card
import torch
from torch import nn


class QLearningNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QLearningNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepQLearningAgent(BridgePlayAgent):
    """
    Agent that contains a network for computing Q-values and uses Q-learning to select actions.
    Note: currently only works for the first player (Defender 1)
    """
    
    def __init__(self, player_type: PlayerType):
        self.q_network = QLearningNetwork(13*8+2, 52)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.training = False
        super().__init__(player_type)

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def get_action(self, observation: PlayObservation) -> Card:
        """
        Select a card from legal actions based on Q-value network.
        
        Args:
            observation: Current game state observation
            
        Returns:
            Card to play
        """

        formatted_observation = self.format_observation(observation)
        q_values = self.q_network(formatted_observation)
        return self.format_response(q_values, observation, formatted_observation)
    
    def format_observation(self, observation: PlayObservation) -> list[float]:
        club_hand = []
        diamond_hand = []
        heart_hand = []
        spade_hand = []
        for card in observation.hand:
            if card.suit == Card.Suit.CLUB:
                club_hand.append(card.rank_value())
            elif card.suit == Card.Suit.DIAMOND:
                diamond_hand.append(card.rank_value())
            elif card.suit == Card.Suit.HEART:
                heart_hand.append(card.rank_value())
            elif card.suit == Card.Suit.SPADE:
                spade_hand.append(card.rank_value())
        club_hand = sorted(club_hand, reverse=True) + [0] * (13 - len(club_hand))
        diamond_hand = sorted(diamond_hand, reverse=True) + [0] * (13 - len(diamond_hand))
        heart_hand = sorted(heart_hand, reverse=True) + [0] * (13 - len(heart_hand))
        spade_hand = sorted(spade_hand, reverse=True) + [0] * (13 - len(spade_hand))

        dummy_club_hand = []
        dummy_diamond_hand = []
        dummy_heart_hand = []
        dummy_spade_hand = []
        for card in observation.dummy_hand:
            if card.suit == Card.Suit.CLUB:
                dummy_club_hand.append(card.rank_value())
            elif card.suit == Card.Suit.DIAMOND:
                dummy_diamond_hand.append(card.rank_value())
            elif card.suit == Card.Suit.HEART:
                dummy_heart_hand.append(card.rank_value())
            elif card.suit == Card.Suit.SPADE:
                dummy_spade_hand.append(card.rank_value())
        dummy_club_hand = sorted(dummy_club_hand, reverse=True) + [0] * (13 - len(dummy_club_hand))
        dummy_diamond_hand = sorted(dummy_diamond_hand, reverse=True) + [0] * (13 - len(dummy_diamond_hand))
        dummy_heart_hand = sorted(dummy_heart_hand, reverse=True) + [0] * (13 - len(dummy_heart_hand))
        dummy_spade_hand = sorted(dummy_spade_hand, reverse=True) + [0] * (13 - len(dummy_spade_hand))

        formatted_observation = club_hand + diamond_hand + heart_hand + spade_hand + dummy_club_hand + dummy_diamond_hand + [observation.contract, observation.tricks_won]

        return torch.tensor(formatted_observation, dtype=torch.float32)
    
    def format_response(self, q_values: torch.Tensor, observation: PlayObservation, formatted_observation: torch.Tensor) -> Card:
        best_card = -1
        best_q_value = -torch.inf
        for card in observation.legal_actions:
            index = dict({'C': 0, 'D': 1, 'H': 2, 'S': 3}).get(card.suit) * 13
            while formatted_observation[index] != card.rank_value():
                index += 1
                if index == 52:
                    raise ValueError("Legal card not found in formatted observation!")
            if q_values[index] > best_q_value:
                best_card = card
                best_q_value = q_values[index]
        return best_card, best_q_value
    
    def feedback(self, observation: PlayObservation, action: Card, reward: float, next_observation: PlayObservation):
        if not self.training:
            return
        self.optimizer.zero_grad()
        formatted_observation = self.format_observation(observation)
        next_formatted_observation = self.format_observation(next_observation)
        q_values = self.q_network(formatted_observation)
        q_value = self.format_response(q_values, observation, formatted_observation)[1]
        target = reward + 0.99 * max(self.q_network(next_formatted_observation) * (next_formatted_observation[:52] > 0))
        loss = (q_value - target)**2
        loss.backward()
        self.optimizer.step()