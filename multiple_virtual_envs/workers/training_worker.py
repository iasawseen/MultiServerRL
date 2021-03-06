import torch
import torch.nn.functional as F
import queue
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def training_worker(args):
    trainer = DQNTrainingWorker(**args)
    trainer.work()


class DQNTrainingWorker:
    def __init__(self, model, target_model, optimizer, replay_queue, batch_size, gamma, target_update=2000):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_queue = replay_queue
        self.memory = ReplayMemory(10000)
        self.batch_size = batch_size
        self.target_update = target_update
        self.updates = 0

    def work(self):
        while True:
            # Get replays from queue
            try:
                # Limit maximum count of gained replays per one optimization
                for _ in range(self.batch_size):
                    observation, action, next_observation, reward = self.replay_queue.get(False)
                    self.memory.push(observation, action, next_observation, reward)
            except queue.Empty:
                # Means that there are no new replays
                pass
            # One optimization step
            self.optimize_model()
            self.updates += 1

            if self.updates % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        #print(list(zip(*transitions))[0])
        batch = Transition(*zip(*transitions))
 
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state))).to(device=self.device)

        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None]).to(device=self.device)
        state_batch = torch.tensor(batch.state).to(device=self.device)
        action_batch = torch.tensor(batch.action).to(device=self.device)
        reward_batch = torch.tensor(batch.reward).to(device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size).to(device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
