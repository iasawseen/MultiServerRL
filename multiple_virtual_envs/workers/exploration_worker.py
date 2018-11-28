import torch
import numpy as np
from itertools import count


def exploration_worker(args):
    worker = ExplorationWorker(**args)
    worker.work()


class ExplorationWorker:
    def __init__(self, env, agent, replay_queue, episodes,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.env = env
        self.agent = agent
        self.replay_queue = replay_queue
        self.episodes = episodes
        self.device = device

    def work(self):
        for _ in range(self.episodes):
            # Reset environment
            state = torch.from_numpy(np.array(self.env.reset())).float().to(self.device)
            total_reward = 0

            for _ in count():
                # Select and perform an action
                action = self.agent.act(state)
                observation, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)
                
                if done:
                    break
                    
                # Observe new state
                next_state = torch.from_numpy(np.array(observation)).float().to(self.device)
                
                # Store the transition in memory
                self.replay_queue.put((state, action, next_state, reward))

                # Move to the next state
                state = next_state
                
            print(total_reward)
