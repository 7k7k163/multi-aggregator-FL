import inspect
import ctypes
import io
import numpy as np
import torch

from model.CNN import CNN_MNIST, CNN_CIFAR
from model.MLP import MLP


def _async_raise(tid, ex_ctypes):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(ex_ctypes):
        ex_ctypes = type(ex_ctypes)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(ex_ctypes))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


def save_model(model, _dir):
    torch.save(model.state_dict(), _dir)
    with open(_dir, 'rb') as f:
        s = f.read()
    return s


def load_model(model, buffer):
    d = torch.load(io.BytesIO(buffer))
    for k in model.state_dict():
        model.state_dict()[k].copy_(d[k])
    return model


def get_model(model_name, type, cuda):
    model = None
    if model_name == 'cnn':
        if type == 'mnist':
            model = CNN_MNIST()
        elif type == 'cifar':
            model = CNN_CIFAR()
    elif model_name == 'mlp':
        if type == 'mnist':
            model = MLP(28 * 28 * 3, 120, 10)
        elif type == 'cifar':
            model = MLP(32 * 32 * 3, 120, 10)
    if cuda:
        model.cuda()
    return model


def FedAvg(w):
    w = [torch.load(io.BytesIO(i)) for i in w]
    w_avg = w[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def make_dict(status, d):
    data_dict = dict()
    if status == 0:
        data_dict['status'] = 0
        data_dict['params'] = {'Bandwidth': np.random.randint(1, 10), 'Mode': ['4G', 'WIFI'][np.random.randint(0, 2)]}
    elif status == 1:
        data_dict['status'] = 1
        data_dict['ip'] = d[0]
        data_dict['port'] = d[1]
    elif status == 2:
        data_dict['status'] = 2
        data_dict['conf'] = d[0]
    elif status == 3:
        data_dict['status'] = 3
        data_dict['serial_number'] = d[0]
        data_dict['epochs'] = d[1]
        data_dict['aggregator'] = d[2]
        data_dict['worker_list'] = d[3]
        data_dict['conf'] = d[4]
        data_dict['model'] = d[5]
    elif status == 4:
        data_dict['status'] = 4
        data_dict['serial_number'] = d[0]
        data_dict['epochs'] = d[1]
        data_dict['aggregator'] = d[2]
        data_dict['worker_list'] = d[3]
        data_dict['conf'] = d[4]
        data_dict['model'] = d[5]
    elif status == 5:
        data_dict['status'] = 5
        data_dict['serial_number'] = d[0]
        data_dict['epochs'] = d[1]
        data_dict['aggregator'] = d[2]
        data_dict['conf'] = d[3]
        data_dict['model'] = d[4]
    elif status == 6:
        data_dict['status'] = 6
        data_dict['acc'] = d[0]
        data_dict['loss'] = d[1]
        # data_dict['model'] = d[2]
    elif status == 7:
        data_dict['status'] = 7
        data_dict['serial_number'] = d[0]
        data_dict['data'] = d[1]
    return data_dict


def dict_to_bytes(data):
    return bytes('{}'.format(data), 'utf-8')


def bytes_to_dict(data):
    return eval(data.decode())
