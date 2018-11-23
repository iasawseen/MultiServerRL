import random
import math
import torch


class RemoteAgent:
    def __init__(self, action_connection, observation_connection):
        self.action_connection = action_connection
        self.obseravtion_connection = observation_connection

    def act(self, observation):
        self.obseravtion_connection.send(observation)
        return self.action_connection.recv()


class Agent:
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        with torch.no_grad():
            return self.model(observation).max(0)[1].view(1, 1)


class AgentWithExploration(Agent):
    def __init__(self, model, eps_end=0.05, eps_start=0.9, eps_decay=200,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(model)
        self.steps_done = 0
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.device = device

    def act(self, observation):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                                       math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            return super().act(observation)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)


def model_worker(real_agent, action_connections, observation_connections):
    while True:
        for i, (act_conn, obs_conn) in enumerate(zip(action_connections, observation_connections)):
            if obs_conn.poll():
                observation = obs_conn.recv()
                action = real_agent.act(observation)
                act_conn.send(action)
