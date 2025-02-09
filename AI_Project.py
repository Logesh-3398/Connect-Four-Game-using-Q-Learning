import numpy as np
import random
import time
import pickle

# Game configuration
row_count = 6
column_count = 7
empty_cell = 0
player_human = 1
player_ai = 2
move_delay = 2.0  
learning_rate = 0.1  # Alpha
discount_factor = 0.9  # Gamma
exploration_rate = 0.1  # Epsilon
reward_win = 100
reward_loss = -100
reward_draw = 10
reward_step = -1

# Q-Table 
try:
    with open("q_table_data.pkl", "rb") as file:
        q_table = pickle.load(file)
except FileNotFoundError:
    q_table = {}

# Game board initialization
def create_board():
    return np.zeros((row_count, column_count), dtype=int)

# Current board state
def show_board(board):
    print("\nGame State:")
    for row in board:
        print(" | ".join([" " if cell == empty_cell else ("X" if cell == player_human else "O") for cell in row]))
    print("-" * (column_count * 4 - 1))

# Valid column moves
def get_valid_columns(board):
    return [col for col in range(column_count) if board[0][col] == empty_cell]

# Placing in row for specified column
def get_open_row(board, column):
    for row in range(row_count - 1, -1, -1):
        if board[row][column] == empty_cell:
            return row
    return None

# Placing in specified position
def drop_disc(board, row, column, player):
    board[row][column] = player

# Checking player has won
def check_winner(board, player):
    # Check horizontal locations
    for row in range(row_count):
        for col in range(column_count - 3):
            if all(board[row, col + i] == player for i in range(4)):
                return True

    # Vertical check
    for col in range(column_count):
        for row in range(row_count - 3):
            if all(board[row + i, col] == player for i in range(4)):
                return True

    # Positive diagonal check
    for row in range(row_count - 3):
        for col in range(column_count - 3):
            if all(board[row + i, col + i] == player for i in range(4)):
                return True

    # Negative diagonal check
    for row in range(3, row_count):
        for col in range(column_count - 3):
            if all(board[row - i, col + i] == player for i in range(4)):
                return True

    return False

# Predict and block opponent moves
def predict_and_block(board, current_player, opponent):
    valid_columns = get_valid_columns(board)

    # Block opponent 
    for col in valid_columns:
        simulated_board = board.copy()
        row = get_open_row(simulated_board, col)
        drop_disc(simulated_board, row, col, opponent)
        if check_winner(simulated_board, opponent):
            return col

    for col in valid_columns:
        simulated_board = board.copy()
        row = get_open_row(simulated_board, col)
        drop_disc(simulated_board, row, col, current_player)
        if check_winner(simulated_board, current_player):
            return col

    return None

# Q-table updation
def update_q_table(state, action, reward, next_state):
    current_q = q_table.get((state, action), 0)
    future_q = max(
        [q_table.get((next_state, next_action), 0) for next_action in get_valid_columns(np.array(next_state))],
        default=0
    )
    new_q = current_q + learning_rate * (reward + discount_factor * future_q - current_q)
    q_table[(state, action)] = new_q

# AI move identification
def decide_ai_move(board, current_player):
    opponent = player_human if current_player == player_ai else player_ai
    valid_columns = get_valid_columns(board)

    # Exploration vs Exploitation
    if random.uniform(0, 1) < exploration_rate:
        return random.choice(valid_columns)

    # Predict,block or executing moves
    block_or_win = predict_and_block(board, current_player, opponent)
    if block_or_win is not None:
        return block_or_win

    # Using Q-table for best move
    best_score = -float("inf")
    best_column = random.choice(valid_columns)
    board_state = tuple(map(tuple, board))

    for col in valid_columns:
        simulated_board = board.copy()
        row = get_open_row(simulated_board, col)
        drop_disc(simulated_board, row, col, current_player)
        future_state = tuple(map(tuple, simulated_board))
        q_value = q_table.get((board_state, col), 0)

        if q_value > best_score:
            best_score = q_value
            best_column = col

    return best_column

# Reward calculation
def calculate_reward(board, current_player, opponent):
    if check_winner(board, current_player):
        return reward_win
    elif check_winner(board, opponent):
        return reward_loss
    elif not get_valid_columns(board):
        return reward_draw
    else:
        return reward_step

# AI vs AI game
def ai_game():
    board = create_board()
    game_running = True
    active_player = player_human

    while game_running:
        show_board(board)
        time.sleep(move_delay)  # Delay 

        valid_columns = get_valid_columns(board)
        if not valid_columns:
            print("The game has ended in a draw")
            break

        board_state = tuple(map(tuple, board))
        selected_column = decide_ai_move(board, active_player)
        row = get_open_row(board, selected_column)
        drop_disc(board, row, selected_column, active_player)

        reward = calculate_reward(board, active_player, player_ai if active_player == player_human else player_human)

        future_state = tuple(map(tuple, board))
        update_q_table(board_state, selected_column, reward, future_state)

        if check_winner(board, active_player):
            show_board(board)
            print(f"\nAI {active_player} wins the game")
            break

        active_player = player_ai if active_player == player_human else player_human

    with open("q_table_data.pkl", "wb") as file:
        pickle.dump(q_table, file)

# User vs AI game
def user_vs_ai_game():
    board = create_board()
    game_running = True
    active_player = player_human

    while game_running:
        show_board(board)

        if active_player == player_human:
            valid_move = False
            while not valid_move:
                try:
                    user_column = int(input(f"\nYour move Choose a column (0-{column_count - 1}): "))
                    if user_column in get_valid_columns(board):
                        valid_move = True
                    else:
                        print("Invalid move Try again")
                except ValueError:
                    print("Invalid input Please enter a number")
            selected_column = user_column
        else:
            print("\nAI is making a move")
            selected_column = decide_ai_move(board, active_player)
            time.sleep(move_delay)  

        row = get_open_row(board, selected_column)
        drop_disc(board, row, selected_column, active_player)

        reward = calculate_reward(board, active_player, player_ai if active_player == player_human else player_human)
        board_state = tuple(map(tuple, board))
        update_q_table(board_state, selected_column, reward, tuple(map(tuple, board)))

        if check_winner(board, active_player):
            show_board(board)
            print(f"\n{'You win' if active_player == player_human else 'AI wins'}")
            break

        if not get_valid_columns(board):
            show_board(board)
            print("\nThe game ends in a draw")
            break

        active_player = player_ai if active_player == player_human else player_human

    with open("q_table_data.pkl", "wb") as file:
        pickle.dump(q_table, file)

# Main menu
print("Select an option:")
print("1- User vs AI")
print("2- AI vs AI")

while True:
    try:
        choice = int(input("Enter your choice (1 or 2): "))
        if choice == 1:
            user_vs_ai_game()
            break
        elif choice == 2:
            ai_game()
            break
        else:
            print("Invalid choice, Please enter 1 or 2")
    except ValueError:
        print("Invalid input, Please enter a number")
