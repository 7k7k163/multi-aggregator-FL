import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import utils, datasets


class FL_trainer:
    def __init__(self, conf, cid):
        self.conf = conf
        self.worker_id = cid
        self.local_model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.train_loader = None
        self.eval_loader = None

    def update(self, conf):
        if conf['model'] != self.conf['model'] or conf['type'] != self.conf['type']:
            self.conf = conf
            self.load_dataset()

    def load_dataset(self):
        # if self.train_dataset is None or self.eval_dataset is None:
        self.train_dataset, self.eval_dataset = datasets.get_dataset("./data/", self.conf["type"])
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        indices = all_range[self.worker_id * data_len: (self.worker_id + 1) * data_len]
        random.shuffle(indices)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.conf["batch_size_train"],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=self.conf['batch_size_test'],
                                      sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                          np.random.choice(range(len(self.eval_dataset)), len(self.eval_dataset))))

    def load_model(self, model=None):
        self.local_model = utils.get_model(self.conf['model'], self.conf['type'], self.conf['cuda'])
        if model is not None:
            self.local_model = utils.load_model(self.local_model, model)

    def local_train(self, model=None):
        if model is None:
            model = self.local_model
        optimiser = torch.optim.Adam(model.parameters(), lr=self.conf['lr'])
        model.train()
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if self.conf['cuda']:
                    data = data.cuda()
                    target = target.cuda()
                optimiser.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimiser.step()
            print(self.worker_id, 'Epoch {:d} done.'.format(e))
        m1, n1 = self.local_eval(model)
        print(self.worker_id, "acc:", m1, ", loss:", n1, " After")
        # utils.save_model(model, './model_file/' + str(self.worker_id) + '.pt')
        return utils.save_model(model, './model_file/worker-' + str(self.conf['model'])
                                + '-' + str(self.worker_id) + '.pt')

    def local_eval(self, model):
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
        return acc, total_l
