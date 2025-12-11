"""Implementation of Pure Monte Carlo Game Search"""

from copy import deepcopy

import numpy as np

from tictactoe import TicTacToe, random_opponent


class Node:
    def __init__(self, id, state, invalid_action, action=None, childs=[]):
        self.id = id
        self.state = state
        self.invalid_action = invalid_action
        self.action = action
        self.childs = childs
        self.value = 0

    def set_value(self, value):
        self.value = value

    def is_leaf(self):
        return not len(self.childs)


class MCTS:
    def __init__(self, opponent=random_opponent, k=5):
        self.k = k
        self.opponent = opponent

        self.n_node = 0

    def run(self):
        results = []
        for i in range(9):
            env = TicTacToe(opponent=self.opponent)
            _ = env.reset()
            state, _, _, info = env.step(i)
            self.root = Node(id=0, state=state, invalid_action=info["invalid_action"])

            self._run(self.root)

            path = self.find_optimal_path(self.root, [])
            results.append([len(path), path[-1].value])

            print(f"\n=> Start from {i}")
            for node in path:
                self.visualize(node)

        for idx, res in enumerate(results):
            print(f"initial pos: {idx}, length: {res[0]}, value: {res[1]}")

    def find_optimal_path(self, node, path):
        path.append(node)
        while not node.is_leaf():
            idx = np.argmax(list(map(lambda x: x.value, node.childs)))
            node = node.childs[idx]
            path.append(node)

        return path

    def _run(self, node):
        node = self.select(node)
        node, done = self.expand(node)
        node = self.play_out(node)

        if done:
            return node

        for child in node.childs:
            self._run(child)

        return node

    def select(self, node):
        if node.is_leaf():
            return node

        idx = np.argmax(list(map(lambda x: x.value, node.childs)))

        return self.select(node.childs[idx])

    def expand(self, node):
        _env = TicTacToe(opponent=self.opponent)

        childs = []
        for i in range(len(node.invalid_action)):
            if not node.invalid_action[i]:  # for all valid actions
                _env.state = deepcopy(node.state)
                state, _, _, info = _env.step(i)
                self.n_node += 1
                childs.append(
                    Node(
                        id=self.n_node,
                        state=state,
                        invalid_action=info["invalid_action"],
                        action=i,
                    )
                )
        node.childs = childs
        return node, True if len(childs) == 0 else False

    def play_out(self, node):
        value = self.simulate(node)
        return self.backpropagate(node, value)

    def simulate(self, node):
        env = TicTacToe(opponent=self.opponent)

        returns = []
        for _ in range(self.k):
            env.state = deepcopy(node.state)
            done = env.is_terminal()
            G_t = 1 if env.is_win(env.state[..., 0]) else 0
            invalid_action = node.invalid_action
            while not done:
                random_action = random_opponent(env.state, invalid_action)
                state, reward, done, info = env.step(random_action)
                invalid_action = info["invalid_action"]
                G_t += reward
            returns.append(G_t)

        return np.mean(returns)

    def backpropagate(self, node: Node, value: float):
        node.set_value(value)

        return node

    def visualize(self, node):
        print(f"Id: {node.id}, action: {node.action}, value: {node.value}")
        print(f"Childs: {[c.id for c in node.childs]}")
        _env = TicTacToe(opponent=self.opponent)
        print(_env.render(node.state))
        print("================================")


def main():
    print("Hello from tic-tac-toe!")

    seed = 0
    np.random.seed(seed)

    mcts = MCTS()
    mcts.run()


if __name__ == "__main__":
    main()
