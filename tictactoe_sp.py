import numpy as np

_player = ["Player2", "Player1"]


class TicTacToeSP:
    def __init__(self):
        self.state = np.zeros((3, 3, 2))  # 3×3 grid for each player and the opponent
        self.turn = 0

    def reset(self, reset_state=None):
        if reset_state is not None:
            self.state = reset_state
            self.turn = 0 if (np.sum(self.state) % 2) == 0 else 1
        else:
            self.state = np.zeros((3, 3, 2))
            self.turn = 0
        return self.state, {"invalid_action": self.state.sum(axis=-1).flatten()}

    def step(self, action):
        x, y = np.unravel_index(action, (3, 3))
        self.state[x, y, self.turn] = 1
        invalid_action = self.state.sum(axis=-1).flatten()
        win = self.is_win(self.state[..., 0])
        lose = self.is_win(self.state[..., 1])
        draw = invalid_action.sum() >= 9

        self.turn = (self.turn + 1) % 2

        if win:
            return (
                self.state,
                1.0,
                True,
                {
                    "invalid_action": np.ones_like(invalid_action),
                    "status": f"{_player[self.turn]} win",
                },
            )
        # elif lose:
        #     return (
        #         self.state,
        #         -1.0,
        #         True,
        #         {
        #             "invalid_action": np.ones_like(invalid_action),
        #             "status": "lose",
        #         },
        #     )
        elif draw:
            return (
                self.state,
                0.0,
                True,
                {"invalid_action": np.ones_like(invalid_action), "status": "draw"},
            )

        return (
            self.state,
            0.0,
            False,
            {"invalid_action": invalid_action, "status": "continue"},
        )

    def is_terminal(self, state=None):
        if state is None:
            state = self.state

        invalid_action = self.state.sum(axis=-1).flatten()
        win = self.is_win(self.state[..., 0])
        lose = self.is_win(self.state[..., 1])
        draw = invalid_action.sum() >= 9

        return win or lose or draw

    def is_win(self, player_grid):
        return np.any(
            (player_grid.sum(axis=0) == 3)
            | (player_grid.sum(axis=1) == 3)
            | (np.diag(player_grid).sum() == 3)
            | (np.diag(player_grid[::-1]).sum() == 3)
        )

    def render(self, state=None):
        if state is None:
            state = self.state
        text = ""
        for xs in state:
            text += "|"
            for x in xs:
                text += "O" if x[..., 0] else ("X" if x[..., 1] else " ")
                text += "|"
            text += "\n"

        return text
