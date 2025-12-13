import numpy as np
from typing import Callable


class TicTacToe:
    def step(self, state: np.ndarray, action: int):
        turn = state.sum() % 2 
        x, y = np.unravel_index(action, (3, 3))
        next_state = state.copy()
        if next_state[x, y].sum() >= 1:
            raise ValueError
        next_state[x, y, turn] = 1
        
        
        invalid_action = next_state.sum(axis=-1).flatten() > 0
        draw = invalid_action.sum() >= 9
        is_win = self.is_win(next_state[..., turn])
        done = draw or is_win
        return next_state, is_win, done, {'invalid_action': invalid_action}

    def is_win(self, player_grid):
        return np.any(
            (player_grid.sum(axis=0) == 3)
            | (player_grid.sum(axis=1) == 3)
            | (np.diag(player_grid).sum() == 3)
            | (np.diag(player_grid[::-1]).sum() == 3)
        )

    def reset(self):
        env_state = np.zeros((3, 3, 2), dtype=np.bool_)
        return env_state, {"invalid_action": env_state.sum(axis=-1).flatten()}

    def render(self, state):
        text = ""
        for xs in state:
            text += "|"
            for x in xs:
                text += "O" if x[..., 0] else ("X" if x[..., 1] else " ")
                text += "|"
            text += "\n"

        return text
    
    
class TicTacToeOpponentWrapper:
    def __init__(self, env: TicTacToe, opponent: Callable[[np.ndarray, np.ndarray], int]):
        self.env = env
        self.opponent = opponent

    def step(self, state: np.ndarray, action: int):
        next_state, is_win, done, info = self.env.step(state, action)
        if done or is_win:
            return next_state, is_win, done, info
        else:
            action = self.opponent(next_state, info['invalid_action'])
            next_state, is_defeat, done, info = self.env.step(next_state, action)
            return next_state, -1 if is_defeat == 1 else 0, done, info
    
    def reset(self, player_first = True):
        if player_first:
            return self.env.reset()
        else:
            state, info = self.env.reset()
            action = self.opponent(state, info['invalid_action'])
            next_state, _, _, info = self.env.step(state, action)
            return next_state, info
    
    def render(self, state: np.ndarray):
        return self.env.render(state)