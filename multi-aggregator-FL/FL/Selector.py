import random


class Selector:
    def __init__(self, worker_list, conf, method='random'):
        self.worker_list = worker_list
        self.conf = conf
        self.method = method

    def select_trainer(self):
        if self.method == 'random':
            return random.sample(self.worker_list, self.conf['d'])

    def select_aggregator(self, trainer_list):
        aggregator_list = list()
        select_list = list()
        num = len(trainer_list)
        if self.method == 'random':
            if num <= 4:
                item = dict()
                item['num'] = [num]
                item['num_of_aggregate'] = 0
                item['aggregators'] = [random.sample(trainer_list, 1)[0].address]
                for i in range(num):
                    aggregator_list.append(item)
            elif 4 < num <= 16:
                for i in range(num // 4 + 1):
                    # select_list.append(len(trainer_list[i * 4: i * 4 + 4]))
                    select_list.append((len(trainer_list[i * 4: i * 4 + 4]), random.sample(trainer_list[i * 4: i * 4 + 4], 1)[0].address))
                addr = random.sample(select_list, 1)[0][1]
                for i in range(num // 4 + 1):
                    for idx in range(select_list[i][0]):
                        item = dict()
                        item['num'] = [select_list[i][0], num // 4 + 1]
                        item['num_of_aggregate'] = 0
                        item['aggregators'] = [select_list[i][1], addr]
                        aggregator_list.append(item)
        return aggregator_list


if __name__ == '__main__':
    selector = Selector([1], [1])
    selector.select_aggregator([1, 2, 3, 4, 5])
