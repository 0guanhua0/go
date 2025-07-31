import argparse
import subprocess
import sys
import time
import os
from itertools import combinations

# sgfmill is used for robust SGF file creation
from sgfmill import sgf

# Local project imports
from game import GoGameState # Uses the provided game logic for internal validation
import config

class Player:
    """Represents a player (Go engine) in the tournament."""
    def __init__(self, name, cmd, elo=1500):
        self.name = name
        self.cmd = cmd.split()
        self.elo = elo
        self.process = None
        self.gtp_log_file = None

    def start(self):
        """Starts the engine process."""
        log_path = f"{self.name}_gtp.log"
        self.gtp_log_file = open(log_path, "w")
        self.process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.gtp_log_file,
            text=True,
            bufsize=1
        )

    def stop(self):
        """Stops the engine process."""
        if self.process:
            try:
                # Nicely ask the engine to quit
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
                # Wait for a moment for a clean shutdown
                self.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, BrokenPipeError):
                # If it doesn't respond, force it to stop
                self.process.kill()
            self.process = None
        if self.gtp_log_file:
            self.gtp_log_file.close()
            self.gtp_log_file = None

    def send_command(self, command):
        """Sends a GTP command to the engine."""
        if not self.process or self.process.poll() is not None:
            print(f"Error: Process for {self.name} is not running.", file=sys.stderr)
            return ""
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

        response = ""
        while True:
            line = self.process.stdout.readline()
            if not line.strip():
                # Empty line signifies end of response in GTP
                break
            response += line
        return response

def parse_gtp_response(response):
    """Parses a GTP response to extract the move or an error."""
    lines = response.strip().split('\n')
    for line in lines:
        if line.startswith('='):
            return line[1:].strip().lower()
        if line.startswith('?'):
            return f"ERROR: {line[1:].strip()}"
    return "ERROR: No valid response"

def update_elo(rating1, rating2, score1):
    """
    Updates Elo ratings for two players based on a game result.
    score1: 1 for player1 win, 0.5 for draw, 0 for player1 loss.
    """
    K = 32 # K-factor, same as used in chess
    expected1 = 1 / (1 + 10**((rating2 - rating1) / 400))
    new_rating1 = rating1 + K * (score1 - expected1)
    # The total change is zero-sum
    new_rating2 = rating2 - (new_rating1 - rating1)
    return round(new_rating1), round(new_rating2)

def run_game(black_player, white_player, game_num, total_games, sgf_dir):
    """
    Manages a single game between two players, returns the winner ('B' or 'W').
    This version uses the sgfmill library for robust SGF generation.
    """
    print(f"\n--- Game {game_num}/{total_games} ---")
    print(f"Black: {black_player.name} ({black_player.elo}), White: {white_player.name} ({white_player.elo})")

    black_player.start()
    white_player.start()

    internal_state = GoGameState(config.BOARD_SIZE)
    GTP_COORD = "ABCDEFGHJKLMNOPQRST"

    # --- SGFmill setup ---
    game = sgf.Sgf_game(size=config.BOARD_SIZE)
    root_node = game.get_root()
    root_node.set('PB', black_player.name)
    root_node.set('PW', white_player.name)
    root_node.set('KM', 7.5)
    root_node.set('RU', 'Chinese')
    current_node = root_node
    winner = 'D' # Default to Draw
    result_string = "0" # SGF result for Draw

    try:
        # Standard GTP setup
        for p in [black_player, white_player]:
            p.send_command(f"boardsize {config.BOARD_SIZE}")
            p.send_command(f"komi 7.5")
            p.send_command("clear_board")

        for turn in range(1, config.BOARD_SIZE * config.BOARD_SIZE * 2 + 1):
            player = black_player if internal_state.get_current_player() == 1 else white_player
            opponent = white_player if internal_state.get_current_player() == 1 else black_player
            color_char = 'B' if internal_state.get_current_player() == 1 else 'W'
            color_lower = color_char.lower()

            command = f"genmove {color_char}"
            response = player.send_command(command)
            move_str = parse_gtp_response(response)

            print(f"Turn {turn}: {player.name} ({color_char}) plays {move_str}")

            if move_str.startswith("ERROR"):
                print(f"Error from {player.name}: {move_str}")
                winner = 'W' if color_char == 'B' else 'B' # Opponent wins on error
                result_string = f"{winner}+F" # Win by Forfeit
                break
            if move_str == "resign":
                winner = 'W' if color_char == 'B' else 'B'
                result_string = f"{winner}+R" # Win by Resignation
                break

            # Convert GTP move to coordinates for sgfmill and internal state
            sgf_move = None # For pass
            if move_str == "pass":
                action = config.BOARD_SIZE * config.BOARD_SIZE
            else:
                try:
                    col = GTP_COORD.find(move_str[0].upper())
                    row = config.BOARD_SIZE - int(move_str[1:])
                    if not (0 <= col < config.BOARD_SIZE and 0 <= row < config.BOARD_SIZE):
                        raise ValueError("Coordinates out of bounds")
                    action = row * config.BOARD_SIZE + col
                    sgf_move = (row, col)
                except (ValueError, IndexError):
                    print(f"Illegal move format from {player.name}: {move_str}")
                    winner = 'W' if color_char == 'B' else 'B'
                    result_string = f"{winner}+F"
                    break

            # Validate and apply move internally
            if action not in internal_state.get_legal_moves():
                print(f"Illegal move by {player.name}: {move_str}")
                winner = 'W' if color_char == 'B' else 'B' # Opponent wins
                result_string = f"{winner}+F"
                break

            # Add the move to the SGF tree
            new_node = current_node.new_child()
            new_node.set_move(color_lower, sgf_move)
            current_node = new_node

            internal_state.apply_move(action)

            # Inform opponent of the move
            opponent.send_command(f"play {color_char} {move_str}")

            game_over, winner_val = internal_state.is_game_over()
            if game_over:
                score = internal_state._get_winner()
                winner = 'B' if score == 1 else 'W'
                # We don't have the exact score from the game state, so we mark win by points.
                result_string = f"{winner}+T" # Win by time/points
                break
        else: # Loop finished without a break (max moves reached)
            winner = 'D'
            result_string = "0"

    finally:
        # Set the final result in the SGF root node
        root_node.set('RE', result_string)

        # Save SGF file
        sgf_filename = os.path.join(sgf_dir, f"{black_player.name}_vs_{white_player.name}_{int(time.time())}.sgf")
        try:
            with open(sgf_filename, "wb") as f:
                f.write(game.serialise())
            print(f"SGF file saved to {sgf_filename}")
        except Exception as e:
            print(f"Error saving SGF file: {e}", file=sys.stderr)

        # Ensure processes are stopped
        black_player.stop()
        white_player.stop()

    return winner

def main():
    parser = argparse.ArgumentParser(description="Round-robin Go tournament manager.")
    parser.add_argument('--player', nargs=2, action='append', metavar=('NAME', 'CMD'),
                        help='Add a player by name and command. Can be used multiple times.')
    parser.add_argument('--games', type=int, default=2, help='Number of games per matchup.')
    parser.add_argument('--sgf_dir', type=str, default='sgf_games', help='Directory to save SGF files.')

    args = parser.parse_args()

    if not args.player or len(args.player) < 2:
        print("Please specify at least two players using --player NAME CMD", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.sgf_dir, exist_ok=True)
    print(f"Saving SGF files to '{args.sgf_dir}'")

    players = [Player(name, cmd) for name, cmd in args.player]

    matchups = list(combinations(players, 2))
    total_games = len(matchups) * args.games

    print(f"--- Starting Tournament ---")
    print(f"Players: {[p.name for p in players]}")
    print(f"Matchups: {len(matchups)}, Games per matchup: {args.games}, Total games: {total_games}")

    game_count = 0
    for p1, p2 in matchups:
        for i in range(args.games):
            game_count += 1
            # Alternate who plays black
            black_player = p1 if i % 2 == 0 else p2
            white_player = p2 if i % 2 == 0 else p1

            winner = run_game(black_player, white_player, game_count, total_games, args.sgf_dir)

            if winner == 'B':
                print(f"Result: Black ({black_player.name}) wins.")
                black_score, white_score = 1.0, 0.0
            elif winner == 'W':
                print(f"Result: White ({white_player.name}) wins.")
                black_score, white_score = 0.0, 1.0
            else: # Draw
                print("Result: Draw.")
                black_score, white_score = 0.5, 0.5

            # Update Elo
            b_elo, w_elo = black_player.elo, white_player.elo
            black_player.elo, white_player.elo = update_elo(b_elo, w_elo, black_score)
            print(f"New Ratings: {black_player.name} = {black_player.elo}, {white_player.name} = {white_player.elo}")


    print("\n--- Tournament Finished ---")
    print("Final Standings:")
    sorted_players = sorted(players, key=lambda p: p.elo, reverse=True)
    for p in sorted_players:
        print(f"  {p.name}: {p.elo}")

if __name__ == "__main__":
    main()

"""
https://katagotraining.org/networks/

kata1-b28c512nbt-s9914646272-d5047971554
2025-07-26 11:37:06 UTC
14055.6 ± 20.2 - (1,626 games)
https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s9914646272-d5047971554.bin.gz

https://raw.githubusercontent.com/lightvector/KataGo/refs/heads/master/cpp/configs/gtp_example.cfg
"""

"""
python tournament.py \
    --player "GoZero" "python engine.py --model_path alphago-zero.safetensors" \
    --player "katago" "katago gtp -model <path_to_your_kata_model.bin.gz> -config <path_to_your_gtp_config.cfg>" \
    --player "gnugo" "gnugo --mode gtp" \
    --games 2
"""
"""
python tournament.py \
    --player "GoZero" "python engine.py --model_path alphago-zero.safetensors" \
    --player "katago" "katago gtp -model kata1-b28c512nbt-s9914646272-d5047971554.bin.gz -config gtp_example.cfg" \
    --player "gnugo" "gnugo --mode gtp" \
    --games 2
"""
