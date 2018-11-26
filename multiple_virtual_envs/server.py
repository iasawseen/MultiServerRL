import argparse
import multiprocessing

from workers.real_env_worker import work

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running server')
    parser.add_argument('--host', type=str, default='localhost', help='')
    parser.add_argument('--port', type=int, default=18000, help='')
    parser.add_argument('--count', type=int, default=2, help='')

    arguments = parser.parse_args()

    processes = []
    try:
        for i in range(arguments.count):
            p = multiprocessing.Process(target=work, args=(arguments.host, arguments.port + i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    finally:
        for p in processes:
            p.terminate()