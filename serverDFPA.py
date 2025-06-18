import copy
import numpy as np
import time
from clientDFPA import *
from collections import defaultdict
from pathlib import Path
import csv
from tqdm import trange


class FedDFPA(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.clients = []
        self.selected_clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.rs_test_acc = []
        self.rs_train_loss = []
        self.times = times
        self.eval_gap = args.eval_gap

        self.args = args
        self._init_clients()
        self.global_headers = copy.deepcopy(
            {k: v for k, v in self.global_model.state_dict().items() if k == 'fc.weight' or k == 'fc.bias'}
        )
        self.global_protos = [None for _ in range(args.num_classes)]

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

    def _init_clients(self):
        for i in range(self.num_clients):
            client = clientDFPA(args=self.args, id=i)
            self.clients.append(client)

    def aggregate_headers(self, client_updates):
        aggregated_params = {}
        for update in client_updates:
            for k, v in update.items():
                if k not in aggregated_params:
                    aggregated_params[k] = v.clone()
                else:
                    aggregated_params[k] += v
        for k in aggregated_params.keys():
            aggregated_params[k] /= len(client_updates)
        self.global_headers = aggregated_params

    def train(self):
        step_iter = trange(self.global_rounds + 1)
        for i in step_iter:
            s_t = time.time()
            self.selected_clients = self.select_clients()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            client_updates = []
            for client in self.selected_clients:
                g_headers = copy.deepcopy(self.global_headers)
                update = client.train(g_headers)
                client_updates.append(update)

            self.aggregate_headers(client_updates)
            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)
            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def send_protos(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.set_protos(self.global_protos)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def select_clients(self):
        join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))
        return selected_clients

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            cl, ct, ns = c.test_metrics()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def train_metrics(self):
        num_samples = []
        losses = []
        tot_correct = []
        for c in self.clients:
            cl, ct, ns = c.train_metrics()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self, acc=None, loss=None):
        save_path = Path("results")
        save_path.mkdir(parents=True, exist_ok=True)
        with open(
                str(save_path / f"bs_{self.args.batch_size}_E{self.args.local_steps}_lr_{self.args.local_learning_rate}_R{self.global_rounds}_N{20}_C{self.join_ratio}_{self.dataset}.csv"),
                'a', newline='') as file:

            mywriter = csv.writer(file, delimiter=',')
            results = []
            stats = self.test_metrics()
            stats_train = self.train_metrics()

            train_loss = sum(stats_train[3]) * 1.0 / sum(stats_train[1])
            train_acc = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
            test_loss = sum(stats[3]) * 1.0 / sum(stats[1])
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])

            if acc is None:
                self.rs_test_acc.append(test_acc)
            else:
                acc.append(test_acc)

            if loss is None:
                self.rs_train_loss.append(train_loss)
            else:
                loss.append(train_loss)

            mean_train_loss = round(train_loss, 4)
            mean_test_loss = round(test_loss, 4)
            mean_train_acc = round(train_acc, 4)
            mean_test_acc = round(test_acc, 4)

            results.append([mean_train_loss, mean_train_acc, mean_test_loss, mean_test_acc])
            mywriter.writerows(results)


def proto_aggregation(local_protos_list):
    agg_protos = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos[label].append(local_protos[label])

    for [label, proto_list] in agg_protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos[label] = proto / len(proto_list)
        else:
            agg_protos[label] = proto_list[0].data

    return agg_protos
