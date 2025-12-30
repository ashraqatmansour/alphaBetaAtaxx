from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, count_disc_count_change

@register_agent("alpha_beta")
class AlphaBeta(Agent):
    """
    A simple agent using minimax with alpha-beta pruning.
    """
    
    def __init__(self):
        super(AlphaBeta, self).__init__()
        self.name = "AlphaBeta"
        self.time_limit = 1.9  # Stay under 2 second limit
        self.my_player = None
    
    def step(self, chess_board, player, opponent):
        """
        Choose the best move using minimax with alphabeta pruning
        """
        start_time = time.time()
        self.start_time = start_time  # Store for time checking in minimax
        self.my_player = player # store player for checking which player we are for move ordering 
        
        valid_moves = get_valid_moves(chess_board, player)
        
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
       
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # player two benifets more from ordered move 
        if player == 2: 
            ordered_moves = self.order_moves(chess_board, valid_moves, player)
        else: 
            ordered_moves = valid_moves
    
        for move in ordered_moves:
        #for move in valid_moves:
            # check if we're running out of time
            if time.time() - start_time > self.time_limit:
                break
            
            board_copy = deepcopy(chess_board)
            execute_move(board_copy, move, player)
            
            score = self.minimax(board_copy, opponent, player, 2, alpha, beta, False)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
        
        time_taken = time.time() - start_time
        #print("My AI's turn took", time_taken, "seconds.")
        
        return best_move
    
    def order_moves(self, chess_board, moves, player):
        """
        Order moves to improve alpha-beta pruning
         prioritize moves that capture more pieces for player 2
         based on hidnt from project slides duplicating is better than moving 
        
        """

        
        total_pieces = np.count_nonzero(chess_board == 1) + np.count_nonzero(chess_board == 2)
        is_early_game = total_pieces < 6
        
        moves_with_scores = []
        
        for move in moves:
            score = 0
            
            captures = count_disc_count_change(chess_board, move, player)
            score += captures * 100
            
            # # Early game esp for  player 1 we reward more expansions 
            # if is_early_game:
            #     r_src, c_src = move.get_src()
            #     r_dest, c_dest = move.get_dest()

            #     # duplication is NOT a 2-tile jump 
            #     if not ((np.abs(r_dest - r_src) == 2) or (np.abs(c_dest - c_src) == 2)):
            #         score += 50
            
            moves_with_scores.append((move, score))
        
        #sort the moves highest to lowest 
        moves_with_scores.sort(key=lambda x: x[1], reverse=True)
        #get the move from the tuple (move, score)
        return [item[0] for item in moves_with_scores]

    def is_terminal(self, chess_board):
        """
        Faster than check_end game because it dont compute scores
        
        """
        # nospaces left
        if np.sum(chess_board == 0) == 0:
            return True
        # 1 player eliminated
        if np.sum(chess_board == 1) == 0 or np.sum(chess_board == 2) == 0:
            return True
        return False
    
    def minimax(self, chess_board, current_player, original_player, depth, alpha, beta, is_maximizing):
        """
        min max tree with alpha beta pruning 
        is_maximizing is true when the max player is playing 
        """
        
        if time.time() - self.start_time > self.time_limit:
            return self.evaluate_board(chess_board, original_player)
        
        # used is_termial faster than check_endgame 
        if depth == 0 or self.is_terminal(chess_board):
            return self.evaluate_board(chess_board, original_player)
        
        valid_moves = get_valid_moves(chess_board, current_player)
        
        # If no valid moves, pass turn to opponent, 3 - current_player so 
        # 3 - 1 or 3 - 2 switching easily 
        # ot is_max at eachn iter making true false or false true 
        if not valid_moves:
            return self.minimax(chess_board, 3 - current_player, original_player, 
                               depth - 1, alpha, beta, not is_maximizing)

        
        if is_maximizing:
            max_eval = float('-inf')

            # player 2 orders their moves for efficiency 
            if self.my_player == 2: 
                ordered_moves = self.order_moves(chess_board, valid_moves, current_player)
            else: 
                ordered_moves = valid_moves
            
            for move in ordered_moves:
            #for move in valid_moves:
                board_copy = deepcopy(chess_board)
                execute_move(board_copy, move, current_player)
                
                eval_score = self.minimax(board_copy, 3 - current_player, original_player,
                                         depth - 1, alpha, beta, False)
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break 
            
            return max_eval
        else:
            
            min_eval = float('inf')

            
            if self.my_player == 2: 
                ordered_moves = self.order_moves(chess_board, valid_moves, current_player)
            else: 
                ordered_moves = valid_moves
            
            for move in ordered_moves:
            #for move in valid_moves:
                board_copy = deepcopy(chess_board)
                execute_move(board_copy, move, current_player)
                
                eval_score = self.minimax(board_copy, 3 - current_player, original_player,
                                         depth - 1, alpha, beta, True)
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break 
            
            return min_eval
    
    def evaluate_board(self, chess_board, player):
        """
        our heauristics 
        """
        
        opponent = 3 - player
        
        
        if self.is_terminal(chess_board):
            # check_end game is slow so only use it when we are at the end 
            is_endgame, p1_score, p2_score = check_endgame(chess_board)
            
            if player == 1:
                score_diff = p1_score - p2_score
            else:
                score_diff = p2_score - p1_score
            
            if score_diff > 0:
                return 10000  # Win
            elif score_diff < 0:
                return -10000  # Loss
            else:
                return -1000 #penalize ties WE WIN OR WE WINNNN 
        
        # if not terminated game use piece difefrences 
        player_count = np.count_nonzero(chess_board == player)
        opponent_count = np.count_nonzero(chess_board == opponent)
        
        return player_count - opponent_count
