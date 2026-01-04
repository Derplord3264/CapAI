import berserk
import chess
import onnxruntime as ort
import numpy as np
import threading
import logging
import math
import time
import sys
import os

API_TOKEN = "<cut>"  
MODEL_FILE = "chess_medium_fp32.onnx"
TIME_LIMIT = 10.0
C_PUCT = 1.0
DECAY = 0.99

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CapAI")

class MCTSNode:
    __slots__ = ['parent', 'children', 'visit_count', 'value_sum', 'prior']
    
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0

class AI:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            if os.path.exists("chess_ai_quantized.onnx"):
                model_path = "chess_ai_quantized.onnx"
            else:
                logger.error(f"Model file {model_path} not found!")
                sys.exit(1)
            
        logger.info(f"Loading ONNX Model: {model_path}...")
        try:
            self.session = ort.InferenceSession(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            sys.exit(1)

    def get_best_move(self, board, think_time=5.0):
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                logger.info(f"Sniper: Found Mate-in-1 ({move.uci()})")
                return move
            board.pop()

        root = MCTSNode()
        self._expand_root(root, board)
        
        start_time = time.time()
        sim_count = 0
        
        while time.time() - start_time < think_time:
            node = root
            sim_board = board.copy()
            path = [node]

            while node.is_expanded():
                node = self._select_child(node)
                if node is None: break
                
                move_to_play = None
                for m, c in path[-1].children.items():
                    if c == node:
                        move_to_play = m
                        break
                sim_board.push(move_to_play)
                path.append(node)

            value = 0.0
            if sim_board.is_game_over():
                value = 1.0 if sim_board.is_checkmate() else 0.0
            else:
                value = self._expand_node(node, sim_board)

            for node in reversed(path):
                node.value_sum += value
                node.visit_count += 1
                value = -value * DECAY 
            
            sim_count += 1

        if not root.children:
             import random
             return random.choice(list(board.legal_moves))

        sorted_moves = sorted(root.children.items(), key=lambda item: item[1].visit_count, reverse=True)
        
        best_valid_move = None
        best_draw_move = None

        for move, child in sorted_moves:
            score = -child.value()
            
            if self.is_suicide(board, move):
                logger.warning(f"Safety: Avoided {move.uci()} (Mate threat)")
                continue 

            board.push(move)
            is_draw = board.can_claim_draw()
            board.pop()

            if is_draw:
                if score > 0.1:
                    logger.warning(f"Killer Instinct: Avoided {move.uci()} (Draw)")
                    if best_draw_move is None: best_draw_move = move
                    continue
                else:
                    logger.info(f"Salvage: {move.uci()} forces draw (Good, we are losing)")
                    return move

            best_valid_move = move
            logger.info(f"Move: {move.uci()} | Score: {score:.3f} | Sims: {sim_count} | Rate: {int(sim_count/think_time)}n/s")
            break
            
        if best_valid_move: return best_valid_move
        if best_draw_move: return best_draw_move
        
        logger.warning("All moves dangerous. Resigning to fate.")
        return sorted_moves[0][0]

    def is_suicide(self, board, move):
        board.push(move)
        if board.is_checkmate() or board.is_game_over():
            board.pop()
            return False
        
        is_dead = False
        for opp_move in board.legal_moves:
            board.push(opp_move)
            if board.is_checkmate():
                is_dead = True
            board.pop()
            if is_dead: break
        
        board.pop()
        return is_dead

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        sqrt_visits = math.sqrt(node.visit_count)

        for child in node.children.values():
            q = -child.value()
            u = C_PUCT * child.prior * sqrt_visits / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand_root(self, node, board):
        p, v = self._predict(board)
        moves = list(board.legal_moves)
        
        noise = np.random.dirichlet([0.3] * len(moves))
        
        clean_p = []
        for move in moves:
            idx = move.from_square * 64 + move.to_square
            clean_p.append(p[idx])
        
        sum_p = sum(clean_p)
        for i, move in enumerate(moves):
            prior = clean_p[i] / sum_p if sum_p > 0 else (1.0 / len(moves))
            prior = 0.75 * prior + 0.25 * noise[i]
            node.children[move] = MCTSNode(node, prior)

    def _expand_node(self, node, board):
        p, v = self._predict(board)
        moves = list(board.legal_moves)
        
        clean_p = []
        for move in moves:
            idx = move.from_square * 64 + move.to_square
            clean_p.append(p[idx])
            
        sum_p = sum(clean_p)
        for i, move in enumerate(moves):
            prior = clean_p[i] / sum_p if sum_p > 0 else (1.0 / len(moves))
            node.children[move] = MCTSNode(node, prior)
        return v

    def _predict(self, board):
        inp = self._board_to_planes(board)
        inp = np.expand_dims(inp, axis=0)
        
        inputs = {self.session.get_inputs()[0].name: inp}
        outs = self.session.run(None, inputs)
        
        policy_logits = outs[0][0]
        value = outs[1][0][0]
        
        exps = np.exp(policy_logits - np.max(policy_logits))
        policy = exps / np.sum(exps)
        return policy, value

    def _board_to_planes(self, board):
        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        planes = np.zeros((14, 8, 8), dtype=np.float32)
        for i, piece in enumerate(pieces):
            for sq in board.pieces(piece, chess.WHITE):
                row, col = divmod(sq, 8)
                planes[i][7-row][col] = 1
            for sq in board.pieces(piece, chess.BLACK):
                row, col = divmod(sq, 8)
                planes[i+6][7-row][col] = 1
        if board.turn == chess.WHITE: planes[12, :, :] = 1
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
            planes[13, :, :] = 1
        return planes

class CapAIBot:
    def __init__(self, token, model_path):
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(self.session)
        try:
            self.user = self.client.account.get()
            self.username = self.user['username']
            logger.info(f"Connected to Lichess as {self.username}")
        except Exception as e:
            logger.error(f"Lichess Connection Failed: {e}")
            sys.exit(1)
        
        self.brain = AI(model_path)

    def start(self):
        logger.info("Listening for challenges...")
        for event in self.client.bots.stream_incoming_events():
            if event['type'] == 'challenge':
                self.handle_challenge(event['challenge'])
            elif event['type'] == 'gameStart':
                game_id = event['game']['id']
                t = threading.Thread(target=self.play_game, args=(game_id,))
                t.start()

    def handle_challenge(self, challenge):
        c_name = challenge['challenger']['name']
        c_id = challenge['id']
        variant = challenge['variant']['key']
        if variant == 'standard':
            logger.info(f"Accepting challenge from {c_name}")
            self.client.bots.accept_challenge(c_id)
        else:
            self.client.bots.decline_challenge(c_id)

    def play_game(self, game_id):
        logger.info(f"--- Game {game_id} Started ---")
        stream = self.client.bots.stream_game_state(game_id)
        board = chess.Board()
        is_white = True
        
        for event in stream:
            if event['type'] == 'gameFull':
                white_id = event['white'].get('id')
                if white_id and white_id.lower() == self.username.lower():
                    is_white = True
                else:
                    is_white = False
                    if 'white' in event and 'name' in event['white']:
                         if event['white']['name'].lower() == self.username.lower():
                             is_white = True

                logger.info(f"Game {game_id}: Playing as {'White' if is_white else 'Black'}")
                
                if event['state']['moves']:
                    for move in event['state']['moves'].split():
                        board.push(chess.Move.from_uci(move))
                
                if self.is_my_turn(board, is_white):
                    self.make_move(game_id, board)

            elif event['type'] == 'gameState':
                board.reset()
                for move in event['moves'].split():
                    board.push(chess.Move.from_uci(move))
                
                if board.is_game_over():
                    logger.info(f"Game {game_id} Over. Result: {board.result()}")
                    break
                
                if self.is_my_turn(board, is_white):
                    self.make_move(game_id, board)

    def is_my_turn(self, board, is_white):
        return (board.turn == chess.WHITE and is_white) or \
               (board.turn == chess.BLACK and not is_white)

    def make_move(self, game_id, board):
        best_move = self.brain.get_best_move(board, think_time=TIME_LIMIT)
        if best_move:
            try:
                self.client.bots.make_move(game_id, best_move.uci())
            except Exception as e:
                logger.error(f"Move Error in {game_id}: {e}")

if __name__ == "__main__":
    bot = CapAIBot(API_TOKEN, MODEL_FILE)
    bot.start()
