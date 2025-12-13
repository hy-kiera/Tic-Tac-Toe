from typing import Dict, Tuple, List
import numpy as np
from tictactoe import TicTacToe
from dataclasses import dataclass

class Node:
    def __init__(self, state, reward, terminated, info):
        self.Q = 0.0
        self.N = 0
        self.win_count = 0
        self.draw_count = 0
        self.lose_count = 0
        self.reward = reward
        self.terminated = terminated
        self.children: Dict[int, "Node"] = {} # key: state
        self.state = state
        self.info = info
    def add_child(self, action: int, node: "Node"):
        self.children[action] = node
        
    def value(self):
        if self.N <= 0:
            return float('-inf')
        else:
            return self.Q / self.N 
        
    def update(self, value):
        self.Q += value
        self.N += 1
        if value >= 0.9999:
            self.win_count += 1
        elif value <= -0.9999:
            self.lose_count += 1
        else:
            self.draw_count += 1
            
        

class MCTS:
    def __init__(self, env: TicTacToe, init_node: Node, epsilon = 0.0, ucb_factor = np.sqrt(2), self_play: bool = True):
        self.nodes: Dict[Tuple, Node] = {}
        self.nodes[tuple(init_node.state.flatten())] = init_node
        self.env = env
        self.self_play = self_play
        self.ucb_factor = ucb_factor
        self.epsilon = epsilon
        
    def explore_action(self, state, info, epsilon, ucb_factor):
        state = tuple(state.flatten())
        valid_action = np.where(~info['invalid_action'])[0]
        valid_action = [a for a in valid_action if a in self.nodes[state].children]
        sum_n = np.sum([self.nodes[state].children[action].N for action in valid_action])
        values = np.array([self.nodes[state].children[action].value() + ucb_factor * np.log(sum_n / self.nodes[state].children[action].N) for action in valid_action])
        argmax_a = valid_action[np.random.choice(np.where(values == values.max())[0])]
        random_a = np.random.choice(valid_action)
        is_random = np.random.random() < epsilon
        return is_random * random_a + (1-is_random) * argmax_a
    
    def select(self, root: Node, path: List = None) -> List[Node]:
        if path is None:
            path = []
        path.append(root)
        valid_action = np.where(~root.info['invalid_action'])[0]
        untried = [a for a in valid_action if a not in root.children]
        if len(untried) > 0:
            return path
        if not root.terminated:
            action = self.explore_action(root.state, root.info, self.epsilon, self.ucb_factor)
            return self.select(root.children[action], path)
        else:
            return path
        
    def expand(self, node: Node) -> List[None]:
        valid_action = np.where(~node.info['invalid_action'])[0]
        untried = [a for a in valid_action if a not in node.children]
        action = np.random.choice(untried)
        next_state, reward, done, next_info = self.env.step(node.state, action)
        key = tuple(next_state.flatten())
        if not key in self.nodes:
            expanded_node = Node(next_state, reward, done, next_info)
            self.nodes[key] = expanded_node
        else:
            expanded_node = self.nodes[key]

        node.add_child(action, expanded_node)
        return expanded_node
                
            
    def simulate(self, node: Node):
        if node.terminated:
            return node.reward
            
        state = node.state
        info = node.info
        valid_action = np.where(~node.info['invalid_action'])[0]
        random_action = np.random.choice(valid_action)
        done = False
        while True:
            state, reward, done, info = self.env.step(state, random_action)
            if done:
                break
            valid_action = np.where(~info['invalid_action'])[0]
            random_action = np.random.choice(valid_action)
        return reward
    
    def backpropagate(self, path, reward, self_play):
        for node in reversed(path):
            node.update(reward)
            if self_play:
                reward = -1 * reward
    
    def run(self, root: Node):
        path = self.select(root)
        leaf = path[-1]
        if not leaf.terminated:
            expanded_node = self.expand(leaf)
            reward = self.simulate(expanded_node)
            self.backpropagate(path + [expanded_node], reward, self_play=self.self_play)
        else:
            expanded_node = leaf
            reward = expanded_node.reward
            self.backpropagate(path, reward, self_play=self.self_play)
        
    def mcts_policy(self, state, info, max_value = True):
        key = tuple(state.flatten())
        valid_action = np.where(~info['invalid_action'])[0]
        random_a = np.random.choice(valid_action)
        
        if key in self.nodes:
            if len(self.nodes[key].children) == 0:
                return random_a      
            tried_actions = np.array([action for action in valid_action if action in self.nodes[key].children])          
            if max_value:
                values = np.array([self.nodes[key].children[action].value() for action in tried_actions])
                argmax_a = valid_action[np.random.choice(np.where(values == values.max())[0])]
                return argmax_a
            else:
                ns = np.array([self.nodes[key].children[action].N for action in tried_actions])
                argmax_a = valid_action[np.random.choice(np.where(ns == ns.max())[0])]
                return argmax_a
        else:
            return random_a

        
