import subprocess
import sgfmill
import sys
import time
import argparse
import shlex  # Import shlex
import os
from sgfmill import sgf, sgf_moves, common
from elo import Rating, rate_1vs1

class GtpProcessPlayer:
    """A wrapper for a Go engine that communicates via the Go Text Protocol (GTP)."""
    def __init__(self, name, command):
        self.name = name

        # The command is now a single string from argparse.
        # We must split it into a list for subprocess.Popen.
        cmd_list = shlex.split(command)
        if not cmd_list:
            raise RuntimeError(f"Engine command for '{self.name}' is empty.")

        try:
            self.process = subprocess.Popen(
                cmd_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True # Ensures cross-platform newline handling
            )
        except FileNotFoundError:
             # Use the first element of the split command list for the error message
             raise RuntimeError(f"Engine command not found for '{self.name}': {cmd_list[0]}")


        # Health check: Give the process a moment to start up or fail.
        time.sleep(1) # Increased for potentially slower-loading engines
        if self.process.poll() is not None:
            # The process terminated prematurely. Read stderr to find out why.
            stderr_output = self.process.stderr.read()
            raise RuntimeError(
                f"Engine '{self.name}' failed to start. "
                f"Return code: {self.process.poll()}\n"
                # Use the original command string for the error message
                f"Command: {command}\n"
                f"Stderr:\n{stderr_output}"
            )

    def send_command(self, command):
        """Sends a GTP command and returns the engine's response."""
        if self.process.poll() is not None:
            sys.stderr.write(f"Cannot send command to '{self.name}', process has terminated.\n")
            return "error"

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

        response = ""
        # Read until the double newline that signifies the end of a GTP response
        while not response.endswith("\n\n"):
            line = self.process.stdout.readline()
            if not line:
                # This can happen if the process dies while we are waiting for a response
                if self.process.poll() is not None:
                    sys.stderr.write(f"Engine '{self.name}' terminated unexpectedly.\n")
                    # Try to get any final error messages
                    stderr_output = self.process.stderr.read()
                    if stderr_output:
                        sys.stderr.write(f"Stderr from {self.name}:\n{stderr_output}\n")
                break
            response += line

        response = response.strip()

        # GTP responses start with '=' for success or '?' for failure
        if response.startswith("="):
            return response[1:].strip()
        else:
            sys.stderr.write(f"Error response from {self.name} for command '{command}': {response}\n")
            return "error"

    def close(self):
        """Closes the engine process."""
        if self.process.poll() is None:
            try:
                self.send_command("quit")
            except BrokenPipeError:
                # Process might have already died, which is fine.
                pass
            self.process.terminate()
            # Wait a moment to ensure it closes
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def play_game(black_player, white_player, board_size, komi):
    """
    Orchestrates a single game between two GTP players.
    Returns the winner ('B' or 'W'), final score, reason, and the SGF game object.
    """
    # --- SGF Setup ---
    game = sgf.Sgf_game(size=board_size)
    root_node = game.get_root()
    root_node.set('PB', black_player.name)
    root_node.set('PW', white_player.name)
    root_node.set('KM', str(komi))
    root_node.set('RU', 'Chinese') # A common default for AI games
    current_sgf_node = root_node

    # --- Engine Setup ---
    for p in [black_player, white_player]:
        p.send_command(f"boardsize {board_size}")
        p.send_command(f"komi {komi}")
        p.send_command("clear_board")

    players = {'B': black_player, 'W': white_player}
    colors = ['B', 'W']
    passes = 0
    turn = 0
    move_log = []

    winner, score_str, reason = None, "N/A", ""

    while turn < (board_size * board_size * 2): # Add move limit to prevent infinite games
        color_char = colors[turn % 2]
        current_player = players[color_char]
        other_player = players[colors[(turn + 1) % 2]]

        move_cmd = f"genmove {color_char.lower()}"
        raw_move = current_player.send_command(move_cmd)

        if raw_move == "error":
            print(f"Turn {turn+1}: {current_player.name} ({color_char}) returned an error. Game over.")
            winner = colors[(turn + 1) % 2]
            reason = "error"
            break

        # Sanitize move: some engines return comments after the move, e.g. 'Q16 # blah'
        # We take only the first part.
        try:
            move = raw_move.split()[0]
        except IndexError:
            print(f"Turn {turn+1}: {current_player.name} ({color_char}) returned an empty move. Game over.")
            winner = colors[(turn + 1) % 2]
            reason = "illegal_move"
            break

        if move.lower() == "resign":
            winner = colors[(turn + 1) % 2]
            reason = "resign"
            break

        # SGF node creation
        new_node = current_sgf_node.new_child()

        try:
            # Convert GTP vertex (e.g., "A19") to SGF coordinates (e.g., (0,0))
            coordinates = common.move_from_vertex(move, board_size)
            if coordinates is None:
                # This handles 'pass', case-insensitively
                passes += 1
                new_node.set(color_char, None)
            else:
                passes = 0
                new_node.set(color_char, coordinates)
        except ValueError:
            print(f"Turn {turn+1}: {current_player.name} ({color_char}) played illegal move (invalid format) {raw_move}. Game over.")
            winner = colors[(turn + 1) % 2]
            reason = "illegal_move"
            break
        current_sgf_node = new_node

        play_cmd = f"play {color_char.lower()} {move}"
        response = other_player.send_command(play_cmd)

        move_log.append(f"Turn {turn+1}: {current_player.name} ({color_char}) plays {move}")
        print(move_log[-1])

        if response == "error":
            print(f"Turn {turn+1}: {current_player.name} ({color_char}) played illegal move {move} (rejected by opponent). Game over.")
            winner = colors[(turn + 1) % 2]
            reason = "illegal_move"
            break

        if passes >= 2:
            score_str = white_player.send_command("final_score")
            print(f"Final Score: {score_str}")
            reason = "score"
            if score_str.upper().startswith('W'):
                winner = 'W'
            elif score_str.upper().startswith('B'):
                winner = 'B'
            else: # Draw or unknown format
                winner = None
                reason = "pass"
            break

        turn += 1

    # --- Finalize SGF and Return ---
    result_sgf_str = ""
    if winner:
        if reason == 'resign':
            result_sgf_str = f"{winner}+R"
        elif reason == 'illegal_move':
            result_sgf_str = f"{winner}+Forfeit"
        elif reason == 'score':
            result_sgf_str = score_str # e.g., "B+10.5"
    elif reason == 'pass': # Draw
        result_sgf_str = "0"

    if result_sgf_str:
        root_node.set('RE', result_sgf_str)
        root_node.set('GC', f"Game {len(move_log)} moves. {reason.capitalize()}.")

    return winner, score_str, reason, game

def run_tournament(args):
    """Main function to run the tournament."""
    try:
        player1 = GtpProcessPlayer(args.p1_name, args.p1_cmd)
        player2 = GtpProcessPlayer(args.p2_name, args.p2_cmd)
    except RuntimeError as e:
        print(f"FATAL: Could not initialize an engine.\n{e}", file=sys.stderr)
        sys.exit(1)

    # --- SGF Directory Setup ---
    if args.sgf_dir:
        try:
            os.makedirs(args.sgf_dir, exist_ok=True)
            print(f"Saving SGF files to '{os.path.abspath(args.sgf_dir)}'")
        except OSError as e:
            print(f"FATAL: Could not create SGF directory '{args.sgf_dir}': {e}", file=sys.stderr)
            sys.exit(1)

    # --- ELO RATING SETUP ---
    p1_rating = Rating(args.p1_rating)
    p2_rating = Rating(args.p2_rating)

    print(f"--- Starting Tournament: {player1.name} vs {player2.name} ---")
    print(f"Initial Ratings: {player1.name} = {p1_rating.rating:.0f}, {player2.name} = {p2_rating.rating:.0f}\n")

    for i in range(args.num_games):
        print(f"\n--- Game {i+1}/{args.num_games} ---")
        if i % 2 == 0:
            black_player, white_player = player1, player2
            black_rating, white_rating = p1_rating, p2_rating
        else:
            black_player, white_player = player2, player1
            black_rating, white_rating = p2_rating, p1_rating

        print(f"Black: {black_player.name} ({black_rating.rating:.0f}), White: {white_player.name} ({white_rating.rating:.0f})")

        try:
            winner, score_str, reason, sgf_game = play_game(
                black_player, white_player, args.board_size, args.komi
            )
        except Exception as e:
            print(f"An error occurred during game {i+1}: {e}", file=sys.stderr)
            print("Aborting tournament.")
            break

        # --- Save SGF ---
        if args.sgf_dir and sgf_game:
            # Sanitize player names for filename
            b_sanitized = "".join(c for c in black_player.name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            w_sanitized = "".join(c for c in white_player.name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
            ts = int(time.time())
            filename = f"{b_sanitized}_vs_{w_sanitized}_{ts}.sgf"
            filepath = os.path.join(args.sgf_dir, filename)
            try:
                with open(filepath, "wb") as f:
                    f.write(sgf_game.serialise())
                print(f"SGF file saved to {filepath}")
            except IOError as e:
                print(f"Error saving SGF file to {filepath}: {e}", file=sys.stderr)

        # --- ELO RATING UPDATE ---
        if winner == 'B':
            print(f"Result: Black ({black_player.name}) wins by {reason}.")
            black_rating, white_rating = rate_1vs1(black_rating, white_rating)
        elif winner == 'W':
            print(f"Result: White ({white_player.name}) wins by {reason}.")
            white_rating, black_rating = rate_1vs1(white_rating, black_rating)
        else:
            print(f"Result: Draw (ended by {reason}).")
            black_rating, white_rating = rate_1vs1(black_rating, white_rating, drawn=True)

        # Update the main rating objects to reflect the new ratings for the next game
        if black_player.name == player1.name:
            p1_rating, p2_rating = black_rating, white_rating
        else:
            p2_rating, p1_rating = black_rating, white_rating

        print(f"New Ratings: {player1.name} = {p1_rating.rating:.0f}, {player2.name} = {p2_rating.rating:.0f}")

    player1.close()
    player2.close()
    print("\n--- Tournament Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Go tournament between two GTP engines.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example - GoZero vs katago:
  python tournament.py \\
    --p1-name "GoZero" --p1-cmd "python engine.py --model_path alphago-zero.safetensors" \\
    --p2-name "katago" --p2-cmd "katago gtp -model kata1-b28c512nbt-s9584861952-d4960414494.bin -config gtp_example.cfg"

Example - katago vs katago:
  python tournament.py \\
    --p1-name "katago" --p1-cmd "katago gtp -model kata1-b28c512nbt-s9584861952-d4960414494.bin -config gtp_example.cfg" \\
    --p2-name "katago" --p2-cmd "katago gtp -model kata1-b28c512nbt-s9584861952-d4960414494.bin -config gtp_example.cfg"

Note: The command for each player (`--p1-cmd`, `--p2-cmd`) must be a single string.
If the command contains spaces, it MUST be enclosed in quotes ("...").
"""
    )
    parser.add_argument("--num_games", type=int, default=2, help="Number of games to play.")
    parser.add_argument("--sgf-dir", type=str, default="sgf_games", help="Directory to save SGF game records. If not specified, defaults to 'sgf_games/'.")
    parser.add_argument("--board-size", type=int, default=19, help="Board size for the games.")
    parser.add_argument("--komi", type=float, default=7.5, help="Komi for the games.")


    # Player 1 arguments
    parser.add_argument("--p1-name", type=str, required=True, help="Name for player 1.")
    parser.add_argument("--p1-cmd", type=str, required=True, help="Command to run player 1 engine (must be quoted).")
    parser.add_argument("--p1-rating", type=int, default=1500, help="Initial ELO rating for player 1.")

    # Player 2 arguments
    parser.add_argument("--p2-name", type=str, required=True, help="Name for player 2.")
    parser.add_argument("--p2-cmd", type=str, required=True, help="Command to run player 2 engine (must be quoted).")
    parser.add_argument("--p2-rating", type=int, default=1500, help="Initial ELO rating for player 2.")

    args = parser.parse_args()
    run_tournament(args)
