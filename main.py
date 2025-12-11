"""Implementation of Pure Monte Carlo Game Search"""

import math
from copy import deepcopy
from collections import namedtuple, defaultdict

import numpy as np

from tictactoe import TicTacToe, random_opponent

_Node = namedtuple("Node", "state invalid_action terminal")


class Node(_Node):
    def find_children(self, opponent):
        if sum(self.invalid_action) == 9:
            return set()  # Terminal node

        env = TicTacToe(opponent)
        children = []
        for i in range(len(self.invalid_action)):
            if not self.invalid_action[i]:
                env.state = deepcopy(np.array(self.state).reshape(3, 3, 2))
                state, _, done, info = env.step(i)
                children.append(
                    Node(
                        state=tuple(state.flatten()),
                        invalid_action=tuple(info["invalid_action"]),
                        terminal=done,
                    )
                )

        return set(children)

    def find_random_chlid(self):
        if sum(self.invalid_action) == 9:
            return None  # Terminal node
        return random_opponent(self.array(self.invalid_action))


class MCTS:
    def __init__(self, opponent=random_opponent, n=50, k=5, c=math.sqrt(2)):
        self.values = defaultdict(int)  # Value of each node
        self.visits = defaultdict(int)  # Visiting count of each node
        self.children = dict()  # Children nodes of each node

        self.n = n  # The number of rollout (policy update)
        self.c = c  # Explroation weight
        self.k = k  # The number of simulation
        self.opponent = opponent  # opponent policy

    def run(self):
        for i in range(9):
            env = TicTacToe(opponent=self.opponent)
            _ = env.reset()
            state, _, done, info = env.step(i)  # initial position is at each cell
            self.root = Node(
                state=tuple(state.flatten()),
                invalid_action=tuple(info["invalid_action"]),
                terminal=done,
            )
            node = self.root
            print(env.render(np.array(node.state).reshape(3, 3, 2)))

            while True:
                for _ in range(self.n):
                    self.rollout(node)
                node = self.choose(node)  # Choose the best successor of node (greedy)

                print(env.render(np.array(node.state).reshape(3, 3, 2)))

                if node.terminal:
                    print(f"Initial pos: {i}, value: {self.values[node]}")
                    print("==========")
                    break

    def choose(self, node):
        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.visits[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.values[n] / self.visits[n]  # average reward

        return max(self.children[node], key=score)

    def rollout(self, node):
        path = self.select(node)
        leaf = path[-1]
        node = self.expand(leaf)
        value = self.simulate(leaf)
        self.backpropagate(path, value)

    def select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _uct_select(self, node):
        assert all(n in self.children for n in self.children[node])

        def uct(n):
            return self.values[n] / self.visits[n] + self.c * math.sqrt(
                math.log(self.visits[node]) / self.visits[n]
            )

        return max(self.children[node], key=uct)

    def expand(self, node):
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children(self.opponent)

    def simulate(self, node):
        env = TicTacToe(opponent=self.opponent)

        returns = []
        for _ in range(self.k):
            env.state = deepcopy(np.array(node.state).reshape(3, 3, 2))
            done = env.is_terminal()
            G_t = 1 if env.is_win(env.state[..., 0]) else 0
            invalid_action = np.array(node.invalid_action)
            while not done:
                random_action = random_opponent(env.state, invalid_action)
                _, reward, done, info = env.step(random_action)
                invalid_action = info["invalid_action"]
                G_t += reward
            returns.append(G_t)

        return np.mean(returns)

    def backpropagate(self, path, value):
        for node in reversed(path):
            self.visits[node] += 1
            self.values[node] += value


def main():
    print("Hello from tic-tac-toe!")

    seed = 0
    np.random.seed(seed)

    mcts = MCTS()
    mcts.run()


if __name__ == "__main__":
    main()
