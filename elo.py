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

def calculate_expected_score(rating1: float, rating2: float) -> float:
    """
    Calculates the expected score for player 1 against player 2.
    This represents the probability of player 1 winning.

    Args:
        rating1 (float): The Elo rating of player 1.
        rating2 (float): The Elo rating of player 2.

    Returns:
        float: The expected score for player 1 (a value between 0 and 1).
    """
    return 1 / (1 + 10**((rating2 - rating1) / 400))

def update_ratings(player1: Rating, player2: Rating, score1: float) -> tuple[Rating, Rating]:
    """
    Updates the Elo ratings for two players based on an actual score.
    This is suitable for batch results, where `score1` can be a win rate.

    Args:
        player1 (Rating): The Rating object for the first player.
        player2 (Rating): The Rating object for the second player.
        score1 (float): The actual score achieved by player 1 (e.g., 1 for a win,
                        0.5 for a draw, 0 for a loss, or a win rate from multiple games).

    Returns:
        tuple[Rating, Rating]: A tuple containing the new Rating objects for player1 and player2.
    """
    # K-factor can be different for each player, but we assume it's the same
    # for a single match. We'll use player1's k_factor.
    k = player1.k_factor

    # Calculate expected score for player 1
    expected_score1 = calculate_expected_score(player1.rating, player2.rating)

    # Calculate the change in rating
    rating_change = k * (score1 - expected_score1)

    # Update ratings
    new_rating1 = player1.rating + rating_change
    # Elo is a zero-sum game, so player2's rating changes by the inverse amount
    new_rating2 = player2.rating - rating_change

    return Rating(new_rating1, player1.k_factor), Rating(new_rating2, player2.k_factor)


def rate_1vs1(player1: Rating, player2: Rating, drawn: bool = False) -> tuple[Rating, Rating]:
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
    # Determine actual score for player 1 based on the game outcome
    if drawn:
        actual_score1 = 0.5
    else: # Player 1 is the winner
        actual_score1 = 1.0

    return update_ratings(player1, player2, actual_score1)


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

    # --- Scenario 4: Update with a win rate (e.g., from 100 games) ---
    print("\n--- Player A wins 60% of games vs Player B ---")
    # Player A (1600) vs Player B (1800)
    # Expected win rate for A is ~24%
    expected_a = calculate_expected_score(player_a.rating, player_b.rating)
    print(f"Expected win rate for A: {expected_a:.2%}")

    # A's actual score is 0.60
    new_a, new_b = update_ratings(player_a, player_b, score1=0.60)
    print(f"New Ratings: Player A = {new_a.rating:.0f}, Player B = {new_b.rating:.0f}")
