import gym
import json
import argparse
import numpy as np
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response


env = gym.make('CartPole-v0').unwrapped


def post_step_request(request):
    json_data = json.loads(request.json_body)
    observation, reward, done, info = env.step(**json_data)

    observation = observation.tolist()

    result = {'observation': observation, 'reward': reward, 'done': done, 'info': info}
    return Response(json=result)


def post_reset_request(request):
    json_data = json.loads(request.json_body)
    observation = env.reset(**json_data)

    observation = observation.tolist()

    result = {'observation': observation}
    return Response(json=result)


def post_render_request(request):
    json_data = json.loads(request.json_body)
    screen = env.render(**json_data)

    screen = screen.tolist()

    result = {'screen': screen}
    return Response(json=result)


def post_close_request(request):
    json_data = json.loads(request.json_body)
    env.close(**json_data)
    result = {'success': True}
    return Response(json=result)


def post_seed_request(request):
    json_data = json.loads(request.json_body)
    env.seed(**json_data)
    result = {'success': True}
    return Response(json=result)


def post_state_request(request):
    json_data = json.loads(request.json_body)
    state = env.state

    if state is not None and isinstance(state, np.ndarray):
        state = state.tolist()

    result = {'state': state}
    return Response(json=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running server')
    parser.add_argument('--host', type=str, default='localhost', help='')
    parser.add_argument('--port', type=int, default=18000, help='')
    arguments = parser.parse_args()

    with Configurator() as config:
        config.add_route('post_step_request', '/post_step_request/')
        config.add_view(post_step_request, route_name='post_step_request', request_method='POST')

        config.add_route('post_reset_request', '/post_reset_request/')
        config.add_view(post_reset_request, route_name='post_reset_request', request_method='POST')

        config.add_route('post_render_request', '/post_render_request/')
        config.add_view(post_render_request, route_name='post_render_request', request_method='POST')

        config.add_route('post_close_request', '/post_close_request/')
        config.add_view(post_close_request, route_name='post_close_request', request_method='POST')

        config.add_route('post_seed_request', '/post_seed_request/')
        config.add_view(post_seed_request, route_name='post_seed_request', request_method='POST')

        config.add_route('post_state_request', '/post_state_request/')
        config.add_view(post_state_request, route_name='post_state_request', request_method='POST')

        app = config.make_wsgi_app()

    server = make_server(arguments.host, arguments.port, app)
    server.serve_forever()
