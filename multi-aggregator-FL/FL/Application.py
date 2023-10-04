import json
import socket

from utils import utils


class Application:
    def __init__(self, path):
        # with open("./config.json", 'r') as f:
        with open(path, 'r') as f:
            self.conf = json.load(f)

    def assignment(self):
        server_socket = socket.create_connection(('127.0.0.1', 20000))
        data_dict = utils.make_dict(2, [self.conf])
        server_socket.sendall(utils.dict_to_bytes(data_dict))
        data = server_socket.recv(1000)
        print(data.decode())
        data = server_socket.recv(1000)
        print(data.decode())
