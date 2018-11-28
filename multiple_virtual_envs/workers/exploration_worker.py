import torch
import numpy as np
from itertools import count


def exploration_worker(args):
    worker = ExplorationWorker(**args)
    worker.work()


class ExplorationWorker:
    def __init__(self, env, agent, replay_queue, episodes):
        self.env = env
        self.agent = agent
        self.replay_queue = replay_queue
        self.episodes = episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def work(self):
        for _ in range(self.episodes):
            # Reset environment
            state = np.array(self.env.reset(), dtype=np.float32)
            total_reward = 0

            for _ in count():
                # Select and perform an action
                action = self.agent.act(state).to(self.device)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                reward = [reward]
                
                if done:
                    break
                    
                # Observe new state
                next_state = np.array(observation, dtype=np.float32)
                
                # Store the transition in memory
                self.replay_queue.put((state, [action], next_state, reward))

                # Move to the next state
                state = next_state
                
            print(total_reward)
