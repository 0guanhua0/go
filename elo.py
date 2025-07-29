import math

class Rating:
    """
    Represents a player's Elo rating.

    Attributes:
        rating (float): The Elo rating of the player.
        k_factor (int): A constant that determines how strongly a result impacts ratings.
                        Higher values mean ratings change more quickly.
    """
    def __init__(self, rating=1500, k_factor=32):
        self.rating = float(rating)
        self.k_factor = k_factor

def rate_1vs1(player1, player2, drawn=False):
    """
    Calculates and returns the new ratings for two players after a single game.
    If not a draw, player1 is assumed to be the winner.

    Args:
        player1 (Rating): The Rating object for the first player (the winner if not a draw).
        player2 (Rating): The Rating object for the second player (the loser if not a draw).
        drawn (bool): True if the game was a draw, False otherwise.

    Returns:
        tuple[Rating, Rating]: A tuple containing the new Rating objects for player1 and player2.
    """
    # Calculate expected scores for each player
    # E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    expected_score1 = 1 / (1 + 10**((player2.rating - player1.rating) / 400))
    expected_score2 = 1 - expected_score1

    # Determine actual scores based on the game outcome
    if drawn:
        actual_score1, actual_score2 = 0.5, 0.5
    else: # Player 1 is the winner
        actual_score1, actual_score2 = 1.0, 0.0

    # Calculate new ratings
    # R'_A = R_A + K * (S_A - E_A)
    new_rating1 = player1.rating + player1.k_factor * (actual_score1 - expected_score1)
    new_rating2 = player2.rating + player2.k_factor * (actual_score2 - expected_score2)

    # Return new Rating objects with the updated values
    return Rating(new_rating1, player1.k_factor), Rating(new_rating2, player2.k_factor)

if __name__ == '__main__':
    # --- Example Usage ---
    # Create two players with initial ratings
    player_a = Rating(rating=1600, k_factor=32)
    player_b = Rating(rating=1800, k_factor=32)

    print(f"Initial Ratings: Player A = {player_a.rating:.0f}, Player B = {player_b.rating:.0f}")

    # --- Scenario 1: Lower-rated player A wins ---
    print("\n--- Player A wins ---")
    new_a, new_b = rate_1vs1(player_a, player_b)
    print(f"New Ratings: Player A = {new_a.rating:.0f}, Player B = {new_b.rating:.0f}")

    # --- Scenario 2: Higher-rated player B wins ---
    print("\n--- Player B wins ---")
    # Note: The winner is always the first argument
    new_b, new_a = rate_1vs1(player_b, player_a)
    print(f"New Ratings: Player A = {new_a.rating:.0f}, Player B = {new_b.rating:.0f}")

    # --- Scenario 3: A draw ---
    print("\n--- Game is a draw ---")
    new_a, new_b = rate_1vs1(player_a, player_b, drawn=True)
    print(f"New Ratings: Player A = {new_a.rating:.0f}, Player B = {new_b.rating:.0f}")
