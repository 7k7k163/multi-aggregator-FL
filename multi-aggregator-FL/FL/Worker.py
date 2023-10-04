import socket
import time

from FL.FL_aggregator import FL_aggregator
from FL.FL_trainer import FL_trainer
from FL.Selector import Selector
from FL.Worker_params import str_to_list, list_to_str
from utils import utils
from threading import Thread, RLock


class Worker:
    def __init__(self, ip, port, buffer=512, backlog=5):
        self.lock = RLock()
        self.initial = False
        self.closed = False
        self.ip = ip
        self.port = port
        self.local_address = None
        self.buffer = buffer
        self.backlog = backlog
        self.serial_number = None
        self.worker_socket = None
        self.worker_list = None
        self.aggregate_list = list()
        self.thread = None
        self.trainer = None
        self.aggregator = None

    def initialize(self):
        self.lock.acquire()
        try:
            if not self.initial:
                worker_socket = socket.create_connection((self.ip, self.port))
                data_dict = utils.make_dict(0, [''])
                data_dict = utils.dict_to_bytes(data_dict)
                worker_socket.sendall(data_dict)
                # data = worker_socket.recv(self.buffer)
                data = b''
                recv_data = worker_socket.recv(self.buffer)
                data += recv_data
                while len(recv_data) == self.buffer:
                    recv_data = worker_socket.recv(self.buffer)
                    data += recv_data
                data = utils.bytes_to_dict(data)
                worker_socket.close()
                self.worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print(data)
                if data['status'] == 1:
                    self.initial = True
                    self.local_address = (data['ip'], data['port'])
                    self.worker_socket.bind(self.local_address)
                    self.worker_socket.listen(self.backlog)
                    self.thread = Worker_listen(self)
                    self.thread.start()
                # print(data)
        finally:
            self.lock.release()

    def send_(self, status, data, model):
        if status == 5:
            addr = data['aggregator']['aggregators'][0]
            data['aggregator']['aggregators'].remove(addr)
            if self.local_address == addr:
                data['model'] = model
                self.aggregate(data)
            else:
                aggregate_socket = socket.create_connection(addr)
                data_dict = utils.make_dict(5, [data['serial_number'], data['epochs'], data['aggregator'], data['conf'], model])
                aggregate_socket.sendall(utils.dict_to_bytes(data_dict))
                aggregate_socket.close()

    def train(self, data):
        if self.trainer is None:
            self.serial_number = data['serial_number']
            self.trainer = FL_trainer(data['conf'], self.serial_number)
            self.trainer.load_dataset()
        else:
            self.trainer.update(data['conf'])
        if data['status'] == 3:
            # print('状态3----------')
            self.worker_list = str_to_list(data['worker_list'])
            self.trainer.load_model(data['model'])
            model = self.trainer.local_train()
            self.send_(5, data, model)
        elif data['status'] == 4:
            # print('状态4----------')
            self.worker_list = str_to_list(data['worker_list'])
            self.trainer.load_model(data['model'])
            model = self.trainer.local_train()
            self.send_(5, data, model)

    def aggregate(self, data):
        if self.aggregator is None:
            self.aggregator = FL_aggregator(data['conf'], self.serial_number)
            self.aggregator.load_dataset()
        else:
            self.aggregator.update(data['conf'])
        print(data['serial_number'], '正在聚合！-----')
        num_of_aggregate = data['aggregator']['num_of_aggregate']
        num_to_aggregate = data['aggregator']['num'][num_of_aggregate]
        self.aggregate_list.append(data['serial_number'])
        if num_of_aggregate == 0:
            self.aggregator.local_update(data['model'], 1)
        else:
            self.aggregator.local_update(data['model'], data['aggregator']['num'][num_of_aggregate-1])
        if num_to_aggregate == len(self.aggregate_list):
            print('聚合测试！', '----', self.serial_number)
            data['aggregator']['num_of_aggregate'] += 1
            self.aggregate_list = list()
            model = self.aggregator.model_aggregate(data['epochs'])
            acc, loss = self.aggregator.model_eval()
            if len(data['aggregator']['num']) != data['aggregator']['num_of_aggregate']:
                data['serial_number'] = self.serial_number
                self.send_(5, data, model)
            else:
                if data['epochs'] < data['conf']['global_epochs']:
                    self.assignment(data['conf'], data['epochs'], model)
                else:
                    worker_socket = socket.create_connection((self.ip, self.port))
                    # data_dict = utils.make_dict(6, [acc, loss, model])
                    data_dict = utils.make_dict(6, [acc, loss])
                    data_dict = utils.dict_to_bytes(data_dict)
                    worker_socket.sendall(data_dict)
                    worker_socket.close()

    def assignment(self, conf, epochs, model):
        selector = Selector(self.worker_list, conf)
        trainer_list = selector.select_trainer()
        aggregator_list = selector.select_aggregator(trainer_list)
        for idx in range(len(trainer_list)):
            worker_socket = socket.create_connection(trainer_list[idx].address)
            data_dict = utils.make_dict(4, [trainer_list[idx].serial_number, epochs+1, aggregator_list[idx],
                                            list_to_str(self.worker_list), conf, model])
            worker_socket.sendall(utils.dict_to_bytes(data_dict))
            worker_socket.close()


class Worker_listen(Thread):
    def __init__(self, agent):
        super(Worker_listen, self).__init__()
        self.worker = agent

    def run(self):
        while self.worker.initial:
            self.worker.lock.acquire()
            # print('开始运行')
            client, address = self.worker.worker_socket.accept()
            data = b''
            recv_data = client.recv(self.worker.buffer)
            data += recv_data
            while len(recv_data) == self.worker.buffer:
                recv_data = client.recv(self.worker.buffer)
                data += recv_data
            data = utils.bytes_to_dict(data)
            client.close()
            if data['status'] == 3:
                # print(data)
                self.worker.train(data)
            elif data['status'] == 4:
                # print(data)
                self.worker.train(data)
            elif data['status'] == 5:
                self.worker.aggregate(data)
            elif data['status'] == 7:
                print(data)
                self.worker.closed = True
            self.worker.lock.release()
            if self.worker.closed:
                break
            time.sleep(1)


if __name__ == '__main__':
    for i in range(8):
        worker = Worker('127.0.0.1', 20000)
        worker.initialize()
