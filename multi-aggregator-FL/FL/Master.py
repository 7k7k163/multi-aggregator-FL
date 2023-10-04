import socket
import time
from threading import Thread, RLock
import utils.utils as utils
from FL.Selector import Selector
from FL.Worker_params import Worker_params, list_to_str


class Master:
    def __init__(self, port, buffer=512, backlog=5):
        self.lock = RLock()
        self.initial = False
        self.closed = False
        self.port = port
        self.buffer = buffer
        self.backlog = backlog
        self.server_socket = None
        self.thread = None
        self.worker_list = None

    def initialize(self):
        self.lock.acquire()
        try:
            if not self.initial:
                self.initial = True
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind(('127.0.0.1', self.port))
                self.server_socket.listen(self.backlog)
                self.thread = Master_listen(self)
                self.worker_list = list()
                self.thread.start()
                print("服务器已打开！")
        finally:
            self.lock.release()

    def assignment(self, conf):
        selector = Selector(self.worker_list, conf)
        trainer_list = selector.select_trainer()
        aggregator_list = selector.select_aggregator(trainer_list)
        model = utils.get_model(conf['model'], conf['type'], conf['cuda'])
        model = utils.save_model(model, './model_file/master-' + str(conf['model']) + '.pt')
        for i in range(len(trainer_list)):
            worker_socket = socket.create_connection(trainer_list[i].address)
            data_dict = utils.make_dict(3, [trainer_list[i].serial_number, 1, aggregator_list[i],
                                            list_to_str(self.worker_list), conf, model])
            worker_socket.sendall(utils.dict_to_bytes(data_dict))
            worker_socket.close()

    def send(self, data_dict, worker_params):
        if worker_params in self.worker_list:
            worker_socket = socket.create_connection(worker_params.address)
            worker_socket.sendall(utils.dict_to_bytes(data_dict))
            # data = worker_socket.recv(self.buffer)
            worker_socket.close()

    def sendToWorkers(self, status, data, worker_list):
        # self.lock.acquire()
        if status == 7:
            for i in range(len(worker_list)):
                data_dict = utils.make_dict(7, [worker_list[i].serial_number, data])
                self.send(data_dict, worker_list[i])
            self.closed = True
        # self.lock.release()


class Master_listen(Thread):
    def __init__(self, agent):
        super(Master_listen, self).__init__()
        self.master = agent
        self.serial_number = 1

    def run(self):
        while self.master.initial:
            self.master.lock.acquire()
            # print('开始运行')
            client, address = self.master.server_socket.accept()
            worker_addr = (address[0], self.master.port + self.serial_number)
            data = b''
            recv_data = client.recv(self.master.buffer)
            data += recv_data
            while len(recv_data) == self.master.buffer:
                recv_data = client.recv(self.master.buffer)
                data += recv_data
            data = utils.bytes_to_dict(data)
            print(data, address)
            if data['status'] == 0:
                worker_params = Worker_params(self.serial_number, worker_addr, data['params'])
                self.serial_number += 1
                self.master.worker_list.append(worker_params)
                data_dict = utils.make_dict(1, worker_addr)
                client.sendall(utils.dict_to_bytes(data_dict))
                client.close()
                # if len(self.master.worker_list) == 10:
                #     print('-------')
                #     self.master.sendToWorkers('conf', self.master.worker_list)
            elif data['status'] == 2:
                self.master.assignment(data['conf'])
                client.sendall(utils.dict_to_bytes('任务开始执行！'))
                app_client = client
            elif data['status'] == 6:
                app_client.sendall(utils.dict_to_bytes('任务完成！acc:{} loss:{}.'.format(data['acc'], data['loss'])))
                self.master.sendToWorkers(7, 'CLOSE', self.master.worker_list)
                app_client.close()
                client.close()
            self.master.lock.release()
            if self.master.closed:
                break
            time.sleep(1)


if __name__ == '__main__':
    master = Master(20000)
    master.initialize()
    # time.sleep(30)
    # master.sendToWorkers('conf', master.worker_list)
