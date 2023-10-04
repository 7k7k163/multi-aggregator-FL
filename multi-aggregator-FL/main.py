import time

from FL.Application import Application
from FL.Master import Master
from FL.Worker import Worker

if __name__ == '__main__':
    master = Master(20000)
    master.initialize()

    for i in range(8):
        worker = Worker('127.0.0.1', 20000)
        worker.initialize()

    time.sleep(5)

    application = Application('./config.json')
    application.assignment()
