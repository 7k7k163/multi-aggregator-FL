import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

from utils import utils, datasets


class FL_aggregator:
    def __init__(self, conf, cid):
        self.conf = conf
        self.id = cid
        self.w_locals = []
        self.global_model = utils.get_model(self.conf['model'], self.conf['type'], self.conf['cuda'])
        self.eval_dataset = None
        self.eval_loader = None
        self.epochs = -1

    def update(self, conf):
        if conf['model'] != self.conf['model'] or conf['type'] != self.conf['type']:
            self.conf = conf
            self.load_dataset()

    def load_dataset(self):
        # if self.train_dataset is None or self.eval_dataset is None:
        train_dataset, self.eval_dataset = datasets.get_dataset("./data/", self.conf["type"])
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.conf['batch_size_test'],
                                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                           np.random.choice(range(len(self.eval_dataset)),
                                                                            len(self.eval_dataset))))

    def local_update(self, model, times):
        # for i in range(times):
        for i in range(times):
            self.w_locals.append(model)

    def model_aggregate(self, epochs):
        self.epochs = epochs
        w_glob = utils.FedAvg(self.w_locals)
        # w_glob = utils.FedAvg(self.w_locals_weight, self.w_locals)
        self.global_model.load_state_dict(w_glob)
        self.w_locals = []
        return utils.save_model(self.global_model, './model_file/' +
                                str(self.conf['model']) + '-' + str(epochs) + '.pt')
        # self.model_eval()
        # for i in range(4):
        #     self.global_model = utils.load_model(self.global_model, self.w_locals[i])
        #     self.model_eval()

    def model_eval(self):
        model = self.global_model
        model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            if self.conf['cuda']:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        print('ID:' + str(self.id) + '  turn [' + str(self.epochs) + ']: acc: ' + str(acc) + ', loss: ' + str(total_l))
        return acc, total_l
