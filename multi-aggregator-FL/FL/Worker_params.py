class Worker_params:
    def __init__(self, serial_number, address, params):
        self.serial_number = serial_number
        self.address = address
        self.params = params

    def __str__(self):
        worker_dict = dict()
        worker_dict['serial_number'] = self.serial_number
        worker_dict['address'] = self.address
        worker_dict['params'] = self.params
        return '{}'.format(worker_dict)


def list_to_str(worker_list):
    list_str = [str(i) for i in worker_list]
    # print(type(list_str))
    # print(eval(list_str[0])['serial_number'])
    return list_str


def str_to_list(list_str):
    worker_list = list()
    for worker in list_str:
        worker = eval(worker)
        worker_list.append(Worker_params(worker['serial_number'], worker['address'], worker['params']))
    return worker_list
