import numpy as np
from mcts import MCTS, Node
from tictactoe import TicTacToe, TicTacToeOpponentWrapper
from opponent import random_opponent, medium_opponent, expert_opponent
import tyro
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pickle
import datetime
@dataclass
class Config:
    ucb_factor: float = np.sqrt(2)
    epsilon: float = 0.0 # epsilon-greedy
    evaluation_frequency: int = 100 
    total_step: int = 10000
    n_eval: int = 100 

def evaluate(eval_env, n, max_value = True):
    rewards = []
    for i in range(n):
        state, info = eval_env.reset()
        done = False
        key = tuple(state.flatten())
        while not done:
            state, reward, done, info = eval_env.step(state, mcts.mcts_policy(state, info, max_value))
            key = tuple(state.flatten())
        rewards.append(reward)
    rewards = np.array(rewards)
    return {
        'win_rate' : np.mean(rewards == 1.0).mean(),
        'draw_rate' : np.mean(rewards == 0.0).mean(),
        'lose_rate' : np.mean(rewards == -1.0).mean()
    }

if __name__ == '__main__':
    config = tyro.cli(Config)
    env = TicTacToe()
    expert_env = TicTacToeOpponentWrapper(env, opponent=expert_opponent)
    medium_env = TicTacToeOpponentWrapper(env, opponent=medium_opponent)
    random_env = TicTacToeOpponentWrapper(env, opponent=random_opponent)
    eval_envs = {
        'expert' : expert_env,
        'medium' : medium_env,
        'random' : random_env
    }
    state, info = env.reset()
    init_positions = [
        tuple(env.step(state, i)[0].flatten()) for i in range(9)
    ]
    root = Node(state, 0, False, info)
    mcts = MCTS(env, root)
    
    logs = defaultdict(list)
    
    for i in tqdm(range(1, config.total_step + 1)):
        mcts.run(root)
        if (i % config.evaluation_frequency) == 0:
            for eval_env in eval_envs:
                for max_value in [True, False]:
                    result = evaluate(eval_envs[eval_env], config.n_eval, max_value)
                    logs[f'{eval_env}_{max_value}'].append(result)
            
            logs['init_position'].append(
                [mcts.nodes[init_position].win_count / mcts.nodes[init_position].N for init_position in init_positions]
            )
    
    
    with open(f'./{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl', 'wb') as f:
        pickle.dump(
            {
                'mcts' : mcts,
                'logs' : logs
            }, f
        )