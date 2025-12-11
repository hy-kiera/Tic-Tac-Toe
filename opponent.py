import numpy as np

def random_opponent(obs, invalid_action):
    valid_action = 1 - invalid_action
    probs = valid_action / valid_action.sum()
    action = np.random.choice(9, 1, p=probs)
    return action


def _medium_opponent(obs, invalid_action, turn=0):
    empty = obs.sum(axis=-1) == 0

    player_board = obs[..., turn]

    for x in range(3):
        if player_board[x].sum() == 2:
            ys = np.where(empty[x])[0]
            if len(ys) > 0:
                y = ys[0]
                return x * 3 + y

    for y in range(3):
        if player_board[:, y].sum() == 2:
            xs = np.where(empty[:, y])[0]
            if len(xs) > 0:
                x = xs[0]
                return x * 3 + y

    diag = np.diag(player_board)
    if diag.sum() == 2:
        for i in range(3):
            if empty[i, i]:
                return i * 3 + i

    anti_diag = np.diag(player_board[::-1])
    if anti_diag.sum() == 2:
        for i in range(3):
            x = 2 - i
            y = i
            if empty[x, y]:
                return x * 3 + y

    return None


def medium_opponent(obs, invalid_action, turn=0):
    action = _medium_opponent(obs, invalid_action, turn)
    if action:
        return action
    else:
        return random_opponent(obs, invalid_action)


def expert_opponent(obs, invalid_action, turn=0):
    action = _medium_opponent(obs, invalid_action, turn)
    if action:
        return action

    action = _medium_opponent(obs, invalid_action, (turn + 1) % 2)
    if action:
        return action
    else:
        return random_opponent(obs, invalid_action)
