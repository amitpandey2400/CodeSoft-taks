"""
Tic-Tac-Toe AI Game Implementation
Uses Minimax algorithm with Alpha-Beta Pruning for optimal AI play
"""

import sys
from typing import List, Tuple, Optional


class TicTacToe:
    """Tic-Tac-Toe game engine with AI player using Minimax algorithm."""
    
    def __init__(self):
        """Initialize the game board and settings."""
        self.board: List[str] = [' ' for _ in range(9)]  # 3x3 board represented as 1D list
        self.human = 'X'
        self.ai = 'O'
        self.empty = ' '
        self.move_count = 0
        
    def print_board(self) -> None:
        """Display the current board state."""
        print("\n")
        for i in range(3):
            print(f" {self.board[i*3]} | {self.board[i*3+1]} | {self.board[i*3+2]} ")
            if i < 2:
                print("-----------")
        print("\n")
        
    def print_positions(self) -> None:
        """Display board position numbers for reference."""
        print("\nPosition reference:")
        for i in range(3):
            print(f" {i*3} | {i*3+1} | {i*3+2} ")
            if i < 2:
                print("-----------")
        print("\n")
        
    def get_available_moves(self) -> List[int]:
        """Return list of available positions on the board."""
        return [i for i in range(9) if self.board[i] == self.empty]
    
    def is_winner(self, player: str) -> bool:
        """Check if the specified player has won."""
        # Define all winning positions
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]               # Diagonals
        ]
        
        for combo in winning_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    
    def is_board_full(self) -> bool:
        """Check if the board is full (draw condition)."""
        return all(cell != self.empty for cell in self.board)
    
    def is_game_over(self) -> bool:
        """Check if the game is over (win or draw)."""
        return self.is_winner(self.human) or self.is_winner(self.ai) or self.is_board_full()
    
    def get_game_status(self) -> str:
        """Return the current game status."""
        if self.is_winner(self.human):
            return "Human"
        elif self.is_winner(self.ai):
            return "AI"
        elif self.is_board_full():
            return "Draw"
        return "Ongoing"
    
    def evaluate_board(self) -> int:
        """
        Evaluate the board for the Minimax algorithm.
        
        Returns:
            10: AI wins
            -10: Human wins
            0: Draw or ongoing
        """
        if self.is_winner(self.ai):
            return 10
        elif self.is_winner(self.human):
            return -10
        else:
            return 0
    
    def minimax(self, depth: int, is_maximizing: bool, 
                alpha: int = float('-inf'), beta: int = float('inf')) -> Tuple[int, Optional[int]]:
        """
        Minimax algorithm with Alpha-Beta Pruning.
        
        Args:
            depth: Current depth in the game tree
            is_maximizing: True if AI turn, False if human turn
            alpha: Best value for maximizer
            beta: Best value for minimizer
            
        Returns:
            Tuple of (score, best_move)
        """
        score = self.evaluate_board()
        
        # Terminal nodes
        if score == 10:  # AI wins
            return score - depth, None
        if score == -10:  # Human wins
            return score + depth, None
        if self.is_board_full():  # Draw
            return 0, None
        
        best_move = None
        
        if is_maximizing:  # AI's turn (maximizing)
            max_eval = float('-inf')
            for move in self.get_available_moves():
                self.board[move] = self.ai
                eval_score, _ = self.minimax(depth + 1, False, alpha, beta)
                self.board[move] = self.empty
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval, best_move
        
        else:  # Human's turn (minimizing)
            min_eval = float('inf')
            for move in self.get_available_moves():
                self.board[move] = self.human
                eval_score, _ = self.minimax(depth + 1, True, alpha, beta)
                self.board[move] = self.empty
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval, best_move
    
    def ai_move(self) -> int:
        """
        Find the best move for the AI using Minimax with Alpha-Beta Pruning.
        
        Returns:
            The position (0-8) where AI should move
        """
        _, best_move = self.minimax(0, True)
        return best_move
    
    def human_move(self) -> bool:
        """
        Get move from human player.
        
        Returns:
            True if move was valid, False otherwise
        """
        while True:
            try:
                move = int(input("Enter your move (0-8): "))
                if move < 0 or move > 8:
                    print("Invalid input! Please enter a number between 0 and 8.")
                    continue
                if self.board[move] != self.empty:
                    print("That position is already taken! Choose another.")
                    continue
                self.board[move] = self.human
                self.move_count += 1
                return True
            except ValueError:
                print("Invalid input! Please enter a valid number.")
    
    def play(self) -> None:
        """Main game loop."""
        print("\n" + "="*50)
        print("Welcome to Tic-Tac-Toe!")
        print("You are X, AI is O")
        print("="*50)
        
        self.print_positions()
        
        # Choose who goes first
        while True:
            first = input("Who goes first? (h for Human, a for AI): ").lower()
            if first in ['h', 'a']:
                ai_first = (first == 'a')
                break
            print("Invalid choice. Please enter 'h' or 'a'.")
        
        print(f"\n{('Human' if not ai_first else 'AI')} goes first!")
        
        while not self.is_game_over():
            self.print_board()
            
            if (self.move_count % 2 == 0 and not ai_first) or (self.move_count % 2 == 1 and ai_first):
                # Human's turn
                print("Your turn!")
                self.human_move()
            else:
                # AI's turn
                print("AI is thinking...")
                move = self.ai_move()
                self.board[move] = self.ai
                self.move_count += 1
                print(f"AI played at position {move}")
        
        # Game over
        self.print_board()
        status = self.get_game_status()
        
        print("="*50)
        if status == "Human":
            print("🎉 Congratulations! You won!")
        elif status == "AI":
            print("🤖 AI wins! Better luck next time!")
        else:
            print("🤝 It's a draw!")
        print("="*50 + "\n")


def main():
    """Main entry point for the game."""
    while True:
        game = TicTacToe()
        game.play()
        
        play_again = input("Play again? (y/n): ").lower()
        if play_again != 'y':
            print("Thanks for playing! Goodbye!")
            break


if __name__ == "__main__":
    main()
