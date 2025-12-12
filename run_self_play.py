"""Implementation of Pure Monte Carlo Game Search"""

import math
from copy import deepcopy
from collections import namedtuple, defaultdict

import numpy as np

from tictactoe_sp import TicTacToeSP
from opponent import random_opponent

_Node = namedtuple("Node", "state invalid_action terminal status")


class Node(_Node):
    def find_children(self):
        if sum(self.invalid_action) == 9:
            return set()  # Terminal node

        env = TicTacToeSP()
        children = []
        for i in range(len(self.invalid_action)):
            if not self.invalid_action[i]:
                _ = env.reset(deepcopy(np.array(self.state).reshape(3, 3, 2)))
                state, _, done, info = env.step(i)
                children.append(
                    Node(
                        state=tuple(state.flatten()),
                        invalid_action=tuple(info["invalid_action"]),
                        terminal=done,
                        status=info["status"],
                    )
                )

        return set(children)

    def find_random_child(self):
        if sum(self.invalid_action) == 9:
            return None  # Terminal node
        env = TicTacToeSP()
        _ = env.reset(deepcopy(np.array(self.state).reshape(3, 3, 2)))
        action = random_opponent(self.state, self.invalid_action)
        state, _, done, info = env.step(action)
        return Node(
            state=tuple(state.flatten()),
            invalid_action=tuple(info["invalid_action"]),
            terminal=done,
            status=info["status"],
        )


class MCTS:
    def __init__(self, n=200, k=5, c=math.sqrt(2)):
        self.values = defaultdict(int)  # Value of each node
        self.visits = defaultdict(int)  # Visiting count of each node
        self.children = dict()  # Children nodes of each node

        self.n = n  # The number of rollout (policy update)
        self.c = c  # Explroation weight
        self.k = k  # The number of simulation

    def run(self):
        print("Start to train MCTS policy")
        env = TicTacToeSP()
        state, info = env.reset()
        root = Node(
            state=tuple(state.flatten()),
            invalid_action=tuple(info["invalid_action"]),
            terminal=env.is_terminal(),
            status=info["status"],
        )
        for i in range(9):
            _ = env.reset()
            state, _, done, info = env.step(i)  # initial position is at each cell
            node = Node(
                state=tuple(state.flatten()),
                invalid_action=tuple(info["invalid_action"]),
                terminal=done,
                status=info["status"],
            )
            self.children[root] = set([node])
            print(env.render(np.array(node.state).reshape(3, 3, 2)))

            while not node.terminal:
                for _ in range(self.n):
                    self.rollout(node)
                node = self.choose(node)  # Choose the best successor of node (greedy)

                print(env.render(np.array(node.state).reshape(3, 3, 2)))
                print(f"value: {self.values[node]}")
                print("-----------------")

            print(f"Initial pos: {i}, status: {node.status}")
            print("==========")

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
        self.children[node] = node.find_children()

    def simulate(self, node):
        env = TicTacToeSP()

        returns = []
        for _ in range(self.k):
            _ = env.reset(deepcopy(np.array(node.state).reshape(3, 3, 2)))
            done = env.is_terminal()
            G_t = 1 if env.is_win(env.state[..., (env.turn + 1) % 2]) else 0
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
            value = -value

    def select_action(self, state, invalid_action, done, status):
        node = Node(
            state=tuple(state.flatten()),
            invalid_action=tuple(invalid_action),
            terminal=done,
            status=status,
        )

        try:
            greedy_node = max(self.children[node], key=lambda n: self.values[n])  # greedy
            action = np.nonzero(greedy_node.invalid_action - invalid_action)[0]
        except KeyError:
            print("Oops! I encountered this state at first, so I'll take a random action:(")
            action = random_opponent(state, invalid_action)

        return action


def play_game(player1, player2):
    env = TicTacToeSP()
    obs, info = env.reset()
    done = False
    done = env.is_terminal()

    print("===============================")
    print("====Let's play Tic-Tac-Toe!====")
    print("===============================")
    print(env.render(obs))
    while not done:
        if env.turn == 0:
            print("=> Player 1 turn")
            if isinstance(player1, str):
                action = int(input("Input (0-8): "))
            else:
                action = player1.select_action(obs, info["invalid_action"], done, info["status"])
        else:
            print("=> Player 2 turn")
            if isinstance(player2, str):
                action = int(input("Input (0-8): "))
            else:
                action = player2.select_action(obs, info["invalid_action"], done, info["status"])
        obs, reward, done, info = env.step(action)
        print(env.render(obs))
    if env.is_win(env.state[..., 0]):
        print("Player 1 win!")
    elif env.is_win(env.state[..., 1]):
        print("Player 2 win!")
    else:
        print("Draw!")


def main():
    print("Hello from tic-tac-toe!")

    seed = 0
    np.random.seed(seed)

    mcts = MCTS()
    mcts.run()

    play_game(mcts, mcts)


if __name__ == "__main__":
    main()
