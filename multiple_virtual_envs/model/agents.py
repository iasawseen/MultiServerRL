import random
import math
import torch


class RemoteAgent:
    """
    Remote agent which sends observations to model worker and returns action
    """
    def __init__(self, action_connection, observation_connection):
        self.action_connection = action_connection
        self.obseravtion_connection = observation_connection

    def act(self, observation):
        self.obseravtion_connection.send(observation)
        return self.action_connection.recv()


class Agent:
    """
    Simply applies model to observation and returns resulting action
    """
    def __init__(self, model):
        self.model = model
        self.device = None

    def act(self, observation):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            return self.model(torch.tensor(observation).to(device=self.device)).max(0)[1].view(1, 1).item()


class AgentWithExploration(Agent):
    """
    Adds some exploration to given model
    """
    def __init__(self, model, eps_end=0.05, eps_start=0.9, eps_decay=200):
        super().__init__(model)
        self.steps_done = 0
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay

    def act(self, observation):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                                       math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            return super().act(observation)
        else:
            return random.randrange(2)


def model_worker(real_agent, action_connections, observation_connections):
    while True:
        for i, (act_conn, obs_conn) in enumerate(zip(action_connections, observation_connections)):
            if obs_conn.poll():
                observation = obs_conn.recv()
                action = real_agent.act(observation)
                act_conn.send(action)
