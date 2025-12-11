import numpy as np


def random_opponent(obs, invalid_action):
    valid_action = 1 - invalid_action
    probs = valid_action / valid_action.sum()
    action = np.random.choice(9, 1, p=probs)
    return action


class TicTacToe:
    def __init__(self, opponent=random_opponent):
        self.state = np.zeros((3, 3, 2))
        self.opponent = opponent

    def step(self, action):
        x, y = np.unravel_index(action, (3, 3))
        self.state[x, y, 0] = 1
        invalid_action = self.state.sum(axis=-1).flatten()
        draw = invalid_action.sum() >= 9
        is_win = self.is_win(self.state[..., 0])

        if is_win:
            return self.state, 1.0, True, {"invalid_action": invalid_action}
        elif draw:
            return self.state, 0.0, True, {"invalid_action": invalid_action}

        opponent_action = self.opponent(self.state, invalid_action)
        x, y = np.unravel_index(opponent_action, (3, 3))
        self.state[x, y, 1] = 1
        enemy_win = self.is_win(self.state[..., 1])
        invalid_action = self.state.sum(axis=-1).flatten()

        if enemy_win:
            return self.state, -1.0, True, {"invalid_action": invalid_action}

        return self.state, 0.0, False, {"invalid_action": invalid_action}

    def is_win(self, player_grid):
        return np.any(
            (player_grid.sum(axis=0) == 3)
            | (player_grid.sum(axis=1) == 3)
            | (np.diag(player_grid).sum() == 3)
            | (np.diag(player_grid[::-1]).sum() == 3)
        )

    def reset(self):
        self.state = np.zeros((3, 3, 2))
        return self.state, {"invalid_action": self.state.sum(axis=-1).flatten()}

    def render(self):
        for xs in self.state:
            print("|", end="")
            for x in xs:
                print("O" if x[..., 0] else ("X" if x[..., 1] else " "), end="|")
            print()
