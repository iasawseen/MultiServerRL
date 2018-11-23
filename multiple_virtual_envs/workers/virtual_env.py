import json
import gym
import requests
from requests.exceptions import RequestException
import time
import numpy as np


class VirtualEnvironment(gym.Env):
    def __init__(self, host_tcp, port_tcp):
        self.host_tcp = host_tcp
        self.port_tcp = port_tcp

    @staticmethod
    def _make_request(request, json_data=None):
        if json_data is None:
            json_data = {}

        flag = True
        res = None
        while flag:
            try:
                res = requests.post(request, json=json_data).json()
            except RequestException:
                time.sleep(1)
                continue
            flag = False
        return res

    def step(self, action):
        json_data = json.dumps({'action': action})
        res = self._make_request('http://{host}:{port}/post_step_request/'.format(host=self.host_tcp,
                                                                                  port=self.port_tcp), json_data)
        return res['observation'], res['reward'], res['done'], res['info']

    def reset(self):
        json_data = json.dumps({})
        res = self._make_request('http://{host}:{port}/post_reset_request/'.format(host=self.host_tcp,
                                                                                   port=self.port_tcp), json_data)
        return res['observation']

    def render(self, mode='human'):
        json_data = json.dumps({'mode': mode})
        res = self._make_request('http://{host}:{port}/post_render_request/'.format(host=self.host_tcp,
                                                                                    port=self.port_tcp), json_data)

        res['screen'] = np.array(res['screen'])
        return res['screen']

    def close(self):
        json_data = json.dumps({})
        res = self._make_request('http://{host}:{port}/post_close_request/'.format(host=self.host_tcp,
                                                                                   port=self.port_tcp), json_data)

    def seed(self, seed=None):
        json_data = json.dumps({'seed': seed})
        res = self._make_request('http://{host}:{port}/post_close_request/'.format(host=self.host_tcp,
                                                                                   port=self.port_tcp), json_data)

    @property
    def state(self):
        json_data = json.dumps({})
        res = self._make_request('http://{host}:{port}/post_state_request/'.format(host=self.host_tcp,
                                                                                   port=self.port_tcp), json_data)
        return res['state']
