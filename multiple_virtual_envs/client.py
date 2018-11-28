import argparse

import torch
from torch import multiprocessing
import torch.optim as optim

from model.agents import AgentWithExploration, model_worker, RemoteAgent
from model.dqn import DQN
from workers.exploration_worker import exploration_worker
from workers.training_worker import training_worker
from workers.virtual_env import VirtualEnvironment

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running client')
    parser.add_argument('--host', type=str, default='localhost', help='')
    parser.add_argument('--port', type=int, default=18000, help='')
    parser.add_argument('--count', type=int, default=2, help='')

    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare model
    model = DQN()
    target_model = DQN()

    model.train()
    target_model.train()

    model.to(device)
    target_model.to(device)

    model.share_memory()
    target_model.share_memory()

    # Create connections between model worker and exploration workers
    action_connections = [multiprocessing.Pipe(duplex=False) for i in range(arguments.count)]
    observation_connections = [multiprocessing.Pipe(duplex=False) for i in range(arguments.count)]
    # Connection between training worker and exploration workers
    replay_queue = multiprocessing.Queue(128)

    join_processes = []
    processes = []
    try:
        # Start model_worker process
        p = multiprocessing.Process(target=model_worker, args=[
            AgentWithExploration(model),
            [conn[1] for conn in action_connections],
            [conn[0] for conn in observation_connections]
        ])
        processes.append(p)
        p.start()

        # Start exploration workers one by one
        for i in range(arguments.count):
            p = multiprocessing.Process(target=exploration_worker, args=[{
                "env": VirtualEnvironment(arguments.host, arguments.port + i),
                "agent": RemoteAgent(action_connections[i][0], observation_connections[i][1]),
                "replay_queue": replay_queue,
                "episodes": 64
            }])
            join_processes.append(p)
            processes.append(p)
            p.start()

        # Training worker
        p = multiprocessing.Process(target=training_worker, args=[{
            "model": model,
            "target_model": target_model,
            "optimizer": optim.Adam(model.parameters()),
            "replay_queue": replay_queue,
            "batch_size": 128,
            "gamma": 0.999
        }])
        processes.append(p)
        p.start()

        for p in join_processes:
            p.join()
    finally:
        for p in processes:
            p.terminate()
