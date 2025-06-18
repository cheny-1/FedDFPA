import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from read_data import read_client_data


class clientDFPA(object):
    def __init__(self, args, id):
        self.dataset = args.dataset
        self.id = id
        self.device = args.device
        self.model = copy.deepcopy(args.model).to(self.device)
        self.criteria = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size
        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()
        self.val_loader = self.load_val_data()
        self.local_steps = args.local_steps
        self.learning_rate = args.local_learning_rate
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        self.local_params = copy.deepcopy(self.model.state_dict())
        self.owned_classes = self._get_owned_classes()
        self._init_parameter_keys()
        self.protos = None
        self.global_protos = None
        self.prev_local_protos = None
        self.loss_mse = nn.MSELoss()
        self.num_classes = args.num_classes

    def _init_parameter_keys(self):
        total_keys = list(self.local_params.keys())
        self.share_keys = [k for k in total_keys if k == 'fc.weight' or k == 'fc.bias']
        self.private_keys = [k for k in total_keys if k not in self.share_keys]

    def _get_owned_classes(self):
        owned = set()
        train_data = read_client_data(self.dataset, self.id, data_type='train')
        for _, y in DataLoader(train_data, batch_size=len(train_data)):
            owned.update(y.unique().cpu().numpy().tolist())
        return list(owned)

    def _class_accuracy(self, model_state_dict):
        temp_model = copy.deepcopy(self.model)
        temp_model.load_state_dict(model_state_dict)
        temp_model.eval()
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred, _ = temp_model(x)
                _, predicted = torch.max(pred, 1)
                for c in self.owned_classes:
                    mask = (y == c)
                    class_correct[c] += (predicted[mask] == c).sum().item()
                    class_total[c] += mask.sum().item()
        return {c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0 for c in self.owned_classes}

    def _compute_alpha(self, global_headers):
        global_state = copy.deepcopy(self.local_params)
        for k in self.share_keys:
            global_state[k] = global_headers[k]

        global_acc = self._class_accuracy(global_state)
        local_acc = self._class_accuracy(self.local_params)

        alphas = {}
        for c in self.owned_classes:
            g_acc = global_acc.get(c, 0)
            l_acc = local_acc.get(c, 0)
            numerator = l_acc - g_acc
            denominator = (g_acc + l_acc + 1e-8)
            ratio = numerator / denominator
            alphas[c] = 1 / (1 + np.exp(-1 * ratio))
        return alphas

    def train(self, global_headers):
        alphas = self._compute_alpha(global_headers)
        mixed_state = copy.deepcopy(self.local_params)
        for k in self.share_keys:
            for c in self.owned_classes:
                mixed_state[k][c] = (self.local_params[k][c] * alphas[c] + global_headers[k][c] * (1 - alphas[c]))

        self.model.load_state_dict(mixed_state, strict=False)
        self.model.train()
        protos = defaultdict(list)
        for step in range(self.local_steps):
            for i, (x, y) in enumerate(self.train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                pred, fea = self.model(x)
                loss = self.criteria(pred, y)

                if self.global_protos is not None and self.prev_local_protos is not None:
                    proto_global = copy.deepcopy(fea.detach())
                    positives = []
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]) and type(self.prev_local_protos[y_c]) != type([]):
                            positives.append((fea[i, :], self.prev_local_protos[y_c]))
                            proto_global[i, :] = self.global_protos[y_c].data

                    if len(positives) > 0:
                        contrastive_loss = 0.0
                        for rep_i, prev_proto in positives:
                            sim = F.cosine_similarity(rep_i.unsqueeze(0), prev_proto.unsqueeze(0))
                            contrastive_loss += (1 - sim)
                        contrastive_loss /= len(positives)
                    else:
                        contrastive_loss = 0.0

                    mse_loss = self.loss_mse(proto_global, fea)
                    loss = 0.5 * contrastive_loss + mse_loss + loss

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(fea[i, :].detach().data)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        self.protos = agg_func(protos)
        self.prev_local_protos = copy.deepcopy(self.protos)
        self.local_params = copy.deepcopy(self.model.state_dict())
        classifier_params = {
            k: v.clone() for k, v in self.local_params.items() if k in ['fc.weight', 'fc.bias']
        }
        return classifier_params

    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, data_type='train')
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, data_type='test')
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def load_val_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        val_data = read_client_data(self.dataset, self.id, data_type='val')
        return DataLoader(val_data, batch_size, drop_last=False, shuffle=False)

    def test_metrics(self, model=None):
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        losses = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.criteria(output[0], y)
                losses += loss.item() * y.shape[0]

                test_acc += (torch.sum(torch.argmax(output[0], dim=1) == y)).item()
                test_num += y.shape[0]

        return losses, test_acc, test_num

    def train_metrics(self, model=None):
        if model == None:
            model = self.model
        model.eval()

        train_acc = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.criteria(output[0], y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_acc += (torch.sum(torch.argmax(output[0], dim=1) == y)).item()

        return losses, train_acc, train_num


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
