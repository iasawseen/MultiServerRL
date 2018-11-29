# MultiServerRL
## About
Here you can find two simple examples of server-client reinforcement learning.
* **simple_virtual_env** shows how to start environment on server and use it by making HTTP requests
* **multiple_virtual_envs** shows how to start more then one environment and use them
## Dependencies
#### Server
`pyramid` - Framework for simple HTTP server creation

`gym` - OpenAI framework with base environments. Here just for example

`numpy` - Mathematics framework

#### Client
`torch` - See https://pytorch.org for details

`torchvision` - See https://pytorch.org for details

`requests` - Framework for making HTTP requests

`numpy` - Mathematics framework

## Run
1. Install dependencies
2. Start server by running `python server.py`
3. Start client by running `python client.py`

By default server and client assumes that host is `localhost` and port is `1800`.

You can pass specific host by using key `--host` and port by using `--port`. 

Additionally in `multiple_virtual_envs` example you can pass count of virtal (client) and real (server) environments running by using `--count`. 
Remember that count of virtual and real environments should be the same. Also you can pass episodes per explorer count by using `--episodes`.

## Structure
### One virtual environments example
#### Server
Server.py starts one HTTP server that running real environment and handle requests.

#### Client
Code in client.py is basic pytorch example copied from [here](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). The only difference is that we use virtual environment instead real.

### Multiple virtual environments example
#### Server
Server.py now creates multiple servers and starts it in separated processes.

#### Client
Client.py runs three different types of processes:
1) **Training worker** handles replays and updates DQN network
2) **Model worker** generates actions by observations using specific agent
3) **Exploration workers** sends actions to environments and receives observations and rewards

#### Files
This is only description of the files contents. Please, see code for details.
* **model** directory contains everything that connected with DQN and agents
    * **model/agents.py** file contains `model_worker` function and DQN wrappers which provide `Actor.act(observation)` interface
    * **model/dqn.py** - simple DQN implementation
* **workers** directory contains all workers except `model_worker`
    * **workers/training_worker.py** file contains DQN trainer. It consumes replays from the replay queue and optimizes DQN parameters
    * **workers/real_env_worker.py** file contains HTTP server worker which runs real CartPole environment and resolves requests
    * **workers/exploration_worker.py** contains worker that receives observations and rewards from environment, puts translation to the replay queue and makes next action which given by agent
    * **workers/virtual_env.py** - virtual environment which provides interface similar to real environment but instead making calculations itself makes HTTP request to corresponding real environment on server
