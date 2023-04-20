import os
import time
import json
import copy
import hashlib
import datetime
import argparse

import psutil
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from dgl import data as dgl_data
from torch_geometric import datasets as pyg_data
from ogb.nodeproppred import NodePropPredDataset
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()
parser.add_argument('method', type=str, default='MLP', help=(
    'MLP | SGC | GCN | IGNN | EIGNN | GIN | SAGE | GAT | GCNII | JKNet'
))
parser.add_argument('dataset', type=str, default='cora', help=(
    'cora | citeseer | pubmed | flickr | arxiv | yelp | reddit | ...'
))
parser.add_argument('--runs', type=int, default=1, help='Default: 1')
parser.add_argument('--gpu', type=int, default=0, help='Default: 0')
parser.add_argument(
    '--split', type=float, default=0,
    help=('Ratio of labels for training.'
          ' Set to 0 to use default split (if any) or 0.6. '
          ' With an integer x the dataset is splitted like Cora with the '
          ' training set be composed by x samples per class. '
          ' Default: 0'))
parser.add_argument(
    '--lr', type=float, default=0.01, help='Learning Rate. Default: 0.01')
parser.add_argument(
    '--dropout', type=float, default=0.0, help='Default: 0')
parser.add_argument('--n-layers', type=int, default=2, help='Default: 2')
parser.add_argument(
    '--weight-decay', type=float, default=0.0, help='Default: 0')
parser.add_argument(
    '--early-stop-epochs', type=int, default=100,
    help='Maximum epochs until stop when accuracy decreasing. Default: 100')
parser.add_argument(
    '--max-epochs', type=int, default=1000,
    help='Maximum epochs. Default: 1000')
parser.add_argument(
    '--hidden', type=int, default=64,
    help='Dimension of hidden representations and implicit state. Default: 64')
parser.add_argument(
    '--heads', type=int, default=1,
    help='Number of attention heads for GAT. Default: 0')
parser.add_argument(
    '--alpha', type=float, default=0.5,
    help='Hyperparameter for GCNII. Default: 0.5')
parser.add_argument(
    '--theta', type=float, default=1.0,
    help='Hyperparameter for GCNII. Default: 1.0')
parser.add_argument(
    '--correct', type=int, default=0,
    help='Iterations for Correct after prediction. Default: 0')
parser.add_argument(
    '--correct-rate', type=float, default=0.1,
    help='Propagation rate for Correct after prediction. Default: 0.1')
parser.add_argument(
    '--smooth', type=int, default=0,
    help='Iterations for Smooth after prediction. Default: 0')
parser.add_argument(
    '--smooth-rate', type=float, default=0.1,
    help='Propagation rate for Smooth after prediction. Default: 0.1')
parser.add_argument(
    '--input-label', type=float, default=0.0,
    help='Ratio of known labels for input. Default: 0')
parser.add_argument(
    '--for-iter', type=int, default=0,
    help='Iterations to produce state in forward-pass. Default: 0')
parser.add_argument(
    '--back-iter', type=int, default=0,
    help='Iterations to accumulate vjp in backward-pass. Default: 0')
parser.add_argument(
    '--drop-state', type=float, default=0.0,
    help='Ratio of state for dropping. Default: 0')
parser.add_argument(
    '--inductive', action='store_true',
    help='Enable the inductive setting')
args = parser.parse_args()

inf = float('inf')
exp = 0.1
norm = lambda x: (x ** 2).mean() ** 0.5

if not torch.cuda.is_available():
    args.gpu = -1
print(datetime.datetime.now(), args)
script_time = time.time()

g_dev = None
gpu = lambda x: x
if args.gpu >= 0:
    g_dev = torch.device('cuda:%d' % args.gpu)
    gpu = lambda x: x.to(g_dev)
coo = torch.sparse_coo_tensor
get_score = lambda y_true, y_pred: f1_score(
    y_true.cpu(), y_pred.cpu(), average='micro').item()


class Optim(object):
    def __init__(self, params):
        self.params = params
        self.opt = torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay)

    def __repr__(self):
        return 'params: %d' % sum(p.numel() for p in self.params)

    def __enter__(self):
        self.opt.zero_grad()
        self.elapsed = time.time()
        return self.opt

    def __exit__(self, *vs, **kvs):
        self.opt.step()
        self.elapsed = time.time() - self.elapsed


class JKNet(nn.Module):
    def __init__(self, din, dout, hidden, n_layers, dropout=0, **kw):
        super(self.__class__, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(gnn.GCNConv(din, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(gnn.GCNConv(hidden, hidden))
        self.lin = nn.Linear(hidden * n_layers, n_labels)
        self.jk = gnn.JumpingKnowledge(mode='cat')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        xs = []
        for conv in self.convs:
            x = F.relu(conv(self.dropout(x), edge_index))
            xs.append(x)
        return self.lin(self.jk(xs))


class GCNII(nn.Module):
    def __init__(
            self, din, dout, hidden, n_layers, dropout=0, **kw):
        super(self.__class__, self).__init__()
        self.lin1 = nn.Linear(din, hidden)
        self.convs = nn.ModuleList([
            gnn.GCN2Conv(
                channels=hidden,
                alpha=kw['alpha'],
                theta=kw['theta'],
                layer=i + 1,
            ) for i in range(n_layers)])
        self.lin2 = nn.Linear(hidden, dout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x0 = x = F.relu(self.lin1(self.dropout(x)))
        for conv in self.convs:
            x = F.relu(conv(self.dropout(x), x0, edge_index))
        return self.lin2(self.dropout(x))


def load_data(name):
    is_bidir = None
    if args.dataset == 'weekday':
        # https://www.mdpi.com/2227-7390/10/8/1262/htm
        fn = 'dataset/weekday.dat'
        if os.path.exists(fn):
            X, Y, E = torch.load(fn)
        else:
            def build_knn_graph(x, base, k=4, b=512, ignore_self=False):
                n = x.shape[0]
                weight = torch.zeros((n, k))
                adj = torch.zeros((n, k), dtype=int)
                for i in range(0, n, b):
                    knn = (
                        (x[i:i+b].unsqueeze(1) - base.unsqueeze(0))
                        .norm(dim=2)
                        .topk(k + int(ignore_self), largest=False))
                    val = knn.values[:, 1:] if ignore_self else knn.values
                    idx = knn.indices[:, 1:] if ignore_self else knn.indices
                    val = torch.softmax(-val, dim=-1)
                    weight[i:i+b] = val
                    adj[i:i+b] = idx
                return weight, adj

            startdate = datetime.date(1980, 1, 1)
            enddate = datetime.date(2020, 1, 1)
            delta = datetime.timedelta(days=1)
            fmt = '%Y%m%d'
            X, Y = [], []
            while startdate < enddate:
                Y.append(startdate.weekday())
                X.append([float(c) for c in startdate.strftime(fmt)])
                startdate += delta
            X = torch.tensor(X)
            Y = torch.tensor(Y, dtype=int)
            n_nodes = X.shape[0]
            _, adj = build_knn_graph(X, X, ignore_self=True)
            src = torch.arange(adj.shape[0]).repeat(adj.shape[1])
            dst = torch.cat([adj[:, i] for i in range(adj.shape[1])], dim=0)
            E = torch.cat((src.unsqueeze(0), dst.unsqueeze(0)))
            torch.save((X, Y, E), 'dataset/weekday.dat')
        train_mask = None
    elif args.dataset.startswith('chain-'):
        """load the synthetic dataset: chain"""
        # https://github.com/SwiftieH/IGNN
        import numpy as np
        import scipy.sparse as sp

        def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            tensor = torch.sparse.FloatTensor(indices, values, shape)
            if device is not None:
                tensor = tensor.to(device)
            return tensor

        r = np.random.RandomState(42)
        c = 10  # num of classes
        n = 20  # chains for each class
        lx = int(args.dataset[6:])  # length of chain
        f = 100  # feature dimension
        n_nodes = c * n * lx
        tn = int(n_nodes * 0.05)  # train nodes
        # NOTE: Our training nodes are fewer than that in EIGNN
        # to keep label information sparse when the chain is long
        tn = c * n
        vl = int(n_nodes * 0.1)  # val nodes
        tt = n_nodes - tn - vl  # test nodes
        ns = 0.00  # noise

        chain_adj = sp.coo_matrix((
            np.ones(lx-1), (np.arange(lx-1), np.arange(1, lx))
        ), shape=(lx, lx))
        # square matrix N = c*n*lx
        adj = sp.block_diag([chain_adj for _ in range(c*n)])

        features = r.uniform(-ns, ns, size=(c, n, lx, f))
        # features = np.zeros_like(features)
        # add class info to the first node of chains.
        features[:, :, 0, :c] += np.eye(c).reshape(c, 1, c)
        features = features.reshape(-1, f)

        labels = np.eye(c).reshape(c, 1, 1, c).repeat(
            n, axis=1).repeat(lx, axis=2)  # one-hot labels
        labels = labels.reshape(-1, c)

        idx_random = np.arange(c*n*lx)
        r.shuffle(idx_random)
        idx_train = idx_random[:tn]
        idx_val = idx_random[tn:tn+vl]
        idx_test = idx_random[tn+vl:tn+vl+tt]

        # porting to pytorch
        features = torch.FloatTensor(np.array(
            features.todense() if sp.issparse(features) else features)).float()
        labels = torch.LongTensor(labels)
        labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        X = features
        Y = labels
        E = adj._indices()
        train_mask = torch.zeros(X.shape[0], dtype=bool)
        valid_mask = torch.zeros(X.shape[0], dtype=bool)
        test_mask = torch.zeros(X.shape[0], dtype=bool)
        train_mask[idx_train] = True
        valid_mask[idx_val] = True
        test_mask[idx_test] = True
        train_masks = [train_mask] * args.runs
        valid_masks = [valid_mask] * args.runs
        test_masks = [test_mask] * args.runs
    elif args.dataset in ('arxiv', 'mag', 'products'):
        ds = NodePropPredDataset(name='ogbn-%s' % args.dataset)
        train_idx, valid_idx, test_idx = map(
            ds.get_idx_split().get, 'train valid test'.split())
        if args.dataset == 'mag':
            train_idx = train_idx['paper']
            valid_idx = valid_idx['paper']
            test_idx = test_idx['paper']
        g, labels = ds[0]
        if args.dataset == 'mag':
            labels = labels['paper']
            g['edge_index'] = g['edge_index_dict'][('paper', 'cites', 'paper')]
            g['node_feat'] = g['node_feat_dict']['paper']
        X = torch.from_numpy(g['node_feat'])
        Y = torch.from_numpy(labels).clone().squeeze(-1)
        E = torch.from_numpy(g['edge_index'])
        n_nodes = X.shape[0]
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True
        is_bidir = False
        train_masks = [train_mask] * args.runs
        valid_masks = [valid_mask] * args.runs
        test_masks = [test_mask] * args.runs
    elif args.dataset == 'ppi':
        g_train, g_valid, g_test = map(
            dgl_data.PPIDataset, 'train valid test'.split())
        n_nodes = 0
        X, Y, E, mask = [], [], [], []
        for mode in 'train valid test'.split():
            for g in dgl_data.PPIDataset(mode):
                X.append(g.ndata['feat'])
                Y.append(g.ndata['label'])
                E.append(n_nodes + torch.cat(
                    [e.view(1, -1) for e in g.edges()], dim=0))
                n_nodes += X[-1].shape[0]
            mask.append(n_nodes)
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        E = torch.cat(E, dim=1)
        train_mask, valid_mask, test_mask = torch.zeros(
            (3, mask[2]), dtype=bool)
        train_mask[:mask[0]] = True
        valid_mask[mask[0]:mask[1]] = True
        test_mask[mask[1]:] = True
        is_bidir = True
        train_masks = [train_mask] * args.runs
        valid_masks = [valid_mask] * args.runs
        test_masks = [test_mask] * args.runs
    elif args.dataset in (
        'cora', 'citeseer', 'pubmed', 'corafull', 'reddit',
        'coauthor-cs', 'coauthor-phy', 'amazon-com', 'amazon-photo',
    ):
        g = (
            dgl_data.CoraGraphDataset() if args.dataset == 'cora'
            else dgl_data.CiteseerGraphDataset() if args.dataset == 'citeseer'
            else dgl_data.PubmedGraphDataset() if args.dataset == 'pubmed'
            else dgl_data.CoraFullDataset() if args.dataset == 'corafull'
            else dgl_data.RedditDataset() if args.dataset == 'reddit'
            else dgl_data.CoauthorCSDataset()
            if args.dataset == 'coauthor-cs'
            else dgl_data.CoauthorPhysicsDataset()
            if args.dataset == 'coauthor-phy'
            else dgl_data.AmazonCoBuyComputerDataset()
            if args.dataset == 'amazon-com'
            else dgl_data.AmazonCoBuyPhotoDataset()
            if args.dataset == 'amazon-photo'
            else None
        )[0]
        X, Y, train_mask, valid_mask, test_mask = map(
            g.ndata.get, 'feat label train_mask val_mask test_mask'.split())
        E = torch.cat([e.view(1, -1) for e in g.edges()], dim=0)
        is_bidir = True
        train_masks = [train_mask] * args.runs
        valid_masks = [valid_mask] * args.runs
        test_masks = [test_mask] * args.runs
    else:
        dn = 'dataset/' + args.dataset
        g = (
            pyg_data.Flickr(dn) if args.dataset == 'flickr'
            else pyg_data.Yelp(dn) if args.dataset == 'yelp'
            else pyg_data.AmazonProducts(dn) if args.dataset == 'amazon'
            else pyg_data.WebKB(dn, args.dataset.capitalize())
            if args.dataset in ('cornell', 'texas', 'wisconsin')
            else pyg_data.WikipediaNetwork(dn, args.dataset)
            if args.dataset in ('chameleon', 'crocodile', 'squirrel')
            else None
        ).data
        X, Y, E, train_mask, valid_mask, test_mask = map(
            g.get, 'x y edge_index train_mask val_mask test_mask'.split())
        if args.dataset in ('flickr', 'yelp', 'amazon'):
            train_masks = [train_mask] * args.runs
            valid_masks = [valid_mask] * args.runs
            test_masks = [test_mask] * args.runs
            is_bidir = True
        else:
            train_masks = [train_mask[:, i % train_mask.shape[1]]
                           for i in range(args.runs)]
            valid_masks = [valid_mask[:, i % train_mask.shape[1]]
                           for i in range(args.runs)]
            test_masks = [test_mask[:, i % train_mask.shape[1]]
                          for i in range(args.runs)]
            is_bidir = False
    if is_bidir is None:
        for i in range(E.shape[1]):
            src, dst = E[:, i]
            if src.item() != dst.item():
                print(src, dst)
                break
        is_bidir = ((E[0] == dst) & (E[1] == src)).any().item()
        print('guess is bidir:', is_bidir)
    n_labels = int(Y.max().item() + 1)
    is_multiclass = len(Y.shape) == 2
    # Save Label Transitional Matrices
    fn = 'dataset/labeltrans/%s.json' % args.dataset
    if not (is_multiclass or os.path.exists(fn)):
        with open(fn, 'w') as file:
            mesh = coo(
                Y[E], torch.ones(E.shape[1]), size=(n_labels, n_labels)
            ).to_dense()
            den = mesh.sum(dim=1, keepdim=True)
            mesh /= den
            mesh[den.squeeze(1) == 0] = 0
            json.dump(mesh.tolist(), file)
    # Remove Self-Loops
    E = E[:, E[0] != E[1]]
    # Get Undirectional Edges
    if not is_bidir:
        E = torch.cat((E, E[[1, 0]]), dim=1)
    if train_mask is None and not args.split:
        args.split = 0.6
    nrange = torch.arange(X.shape[0])
    if 0 < args.split < 1:
        torch.manual_seed(42)  # the answer
        train_masks, valid_masks, test_masks = [], [], []
        for _ in range(args.runs):
            train_mask = torch.zeros(X.shape[0], dtype=bool)
            valid_mask = torch.zeros(X.shape[0], dtype=bool)
            test_mask = torch.zeros(X.shape[0], dtype=bool)
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
            if is_multiclass:
                val_num = test_num = int((1 - args.split) / 2 * X.shape[0])
                idx = torch.randperm(X.shape[0])
                train_mask[idx[val_num + test_num:]] = True
                valid_mask[idx[:val_num]] = True
                test_mask[idx[val_num:val_num + test_num]] = True
            else:
                for c in range(n_labels):
                    label_idx = nrange[Y == c]
                    val_num = test_num = int(
                        (1 - args.split) / 2 * label_idx.shape[0])
                    perm = label_idx[torch.randperm(label_idx.shape[0])]
                    train_mask[perm[val_num + test_num:]] = True
                    valid_mask[perm[:val_num]] = True
                    test_mask[perm[val_num:val_num + test_num]] = True
    elif int(args.split):
        # NOTE: work only for graphs with single labelled nodes.
        torch.manual_seed(42)  # the answer
        train_masks, valid_masks, test_masks = [], [], []
        for _ in range(args.runs):
            train_mask = torch.zeros(X.shape[0], dtype=bool)
            for y in range(n_labels):
                label_mask = Y == y
                train_mask[
                    nrange[label_mask][
                        torch.randperm(label_mask.sum())[:int(args.split)]]
                ] = True
            valid_mask = ~train_mask
            valid_mask[
                nrange[valid_mask][torch.randperm(valid_mask.sum())[500:]]
            ] = False
            test_mask = ~(train_mask | valid_mask)
            test_mask[
                nrange[test_mask][torch.randperm(test_mask.sum())[1000:]]
            ] = False
            train_masks.append(train_mask)
            valid_masks.append(valid_mask)
            test_masks.append(test_mask)
    return X, Y, E, train_masks, valid_masks, test_masks, is_bidir


class Stat(object):
    def __init__(self):
        self.preprocess_time = 0
        self.training_times = []
        self.evaluation_times = []

        self.best_test_scores = []
        self.best_times = []
        self.best_training_times = []

        self.mem = psutil.Process().memory_info().rss / 1024 / 1024
        self.gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            self.gpu = torch.cuda.memory_allocated(g_dev) / 1024 / 1024

    def start_preprocessing(self):
        self.preprocess_time = time.time()

    def stop_preprocessing(self):
        self.preprocess_time = time.time() - self.preprocess_time

    def start_run(self):
        self.params = None
        self.scores = []
        self.acc_training_times = []
        self.acc_times = []
        self.training_times.append(0.)
        self.evaluation_times.append(0.)

    def record_training(self, elapsed):
        self.training_times[-1] += elapsed

    def record_evaluation(self, elapsed):
        self.evaluation_times[-1] += elapsed

    def evaluate_result(self, y):
        valid_y = y[valid_mask]
        test_y = y[test_mask]
        if is_multiclass:
            self.scores.append([
                get_score(Y[valid_mask], valid_y > 0.5),
                get_score(Y[test_mask], test_y > 0.5)])
        else:
            self.scores.append([
                get_score(Y[valid_mask], valid_y.argmax(dim=1)),
                get_score(Y[test_mask], test_y.argmax(dim=1))])
        self.acc_training_times.append(self.training_times[-1])
        self.acc_times.append(self.preprocess_time + self.training_times[-1])
        dec_epochs = len(self.scores) - 1 - torch.tensor(
            self.scores).max(dim=0).indices[0]
        if dec_epochs == 0:
            self.best_acc = self.scores[-1][0]
            self.best_y = y
        return dec_epochs >= args.early_stop_epochs

    def end_run(self):
        print('val scores:', [s for s, _ in self.scores])
        print('test scores:', [s for _, s in self.scores])
        print('acc training times:', self.acc_training_times)
        self.scores = torch.tensor(self.scores)
        print('max scores:', self.scores.max(dim=0).values)
        idx = self.scores.max(dim=0).indices[0]
        self.best_test_scores.append((idx, self.scores[idx, 1]))
        self.best_training_times.append(self.acc_training_times[idx])
        self.best_times.append(self.acc_times[idx])
        print('best test score:', self.best_test_scores[-1])

    def end_all(self):
        conv = 1.0 + torch.tensor([
            idx for idx, _ in self.best_test_scores])
        score = 100 * torch.tensor([
            score for _, score in self.best_test_scores])
        tm = torch.tensor(self.best_times)
        ttm = torch.tensor(self.best_training_times)
        print('converge time: %.3f±%.3f' % (
            tm.mean().item(), tm.std().item()))
        print('converge training time: %.3f±%.3f' % (
            ttm.mean().item(), ttm.std().item()))
        print('converge epochs: %.3f±%.3f' % (
            conv.mean().item(), conv.std().item()))
        print('score: %.2f±%.2f' % (score.mean().item(), score.std().item()))

        # Output Used Time
        print('preprocessing time: %.3f' % self.preprocess_time)
        for name, times in (
            ('total training', self.training_times),
            ('total evaluation', self.evaluation_times),
        ):
            times = torch.tensor(times or [0], dtype=float)
            print('%s time: %.3f±%.3f' % (
                name, times.mean().item(), times.std().item()))

        # Output Used Space
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        gpu = 0
        if g_dev is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            gpu = torch.cuda.max_memory_allocated(g_dev) / 1024 / 1024
        print('pre_memory: %.2fM + %.2fM = %.2fM' % (
            self.mem, self.gpu, self.mem + self.gpu))
        print('max_memory: %.2fM + %.2fM = %.2fM' % (
            mem, gpu, mem + gpu))
        print('memory_diff: %.2fM + %.2fM = %.2fM' % (
            mem - self.mem,
            gpu - self.gpu,
            mem + gpu - self.mem - self.gpu))


X, Y, E, train_masks, valid_masks, test_masks, is_bidir = load_data(
    args.dataset)
n_nodes = X.shape[0]
n_features = X.shape[1]
is_multiclass = len(Y.shape) == 2
n_labels = Y.shape[1] if is_multiclass else int(Y.max().item() + 1)
deg = E.shape[1] / n_nodes
print('nodes: %d' % n_nodes)
print('features: %d' % n_features)
print('classes: %d' % n_labels)
print('is_multiclass:', is_multiclass)
print('edges without self-loops: %d' % ((E.shape[1] - n_nodes) / 2))
print('average degree: %.2f' % deg)
train_sum = sum([m.sum() for m in train_masks]) / len(train_masks)
valid_sum = sum([m.sum() for m in valid_masks]) / len(valid_masks)
test_sum = sum([m.sum() for m in test_masks]) / len(test_masks)
print('split: %d (%.2f%%) / %d (%.2f%%) / %d (%.2f%%)' % (
    train_sum, 100 * train_sum / n_nodes,
    valid_sum, 100 * valid_sum / n_nodes,
    test_sum, 100 * test_sum / n_nodes,
))
print('intra_rate: %.2f%%' % (100 * (
    Y[E[0]] == Y[E[1]]).sum().float() / E.shape[1] / (
        n_labels if is_multiclass else 1)))

if is_multiclass:
    _cri = nn.BCEWithLogitsLoss(reduction='none')
    criterion = lambda x, y: _cri(x, y).sum(dim=1)
    sg = torch.sigmoid
    lsg = F.logsigmoid
    thres_for = 1e-4 * 0.5
    thres_back = 1e-5 * 0.25
else:
    criterion = lambda x, y: F.cross_entropy(x, y, reduction='none')
    sg = lambda x: torch.softmax(x, dim=-1)
    lsg = lambda x: F.log_softmax(x, dim=-1)
    thres_for = 1e-4 / n_labels
    thres_back = 1e-5 * (n_labels - 1) / n_labels / n_labels


def norm_adj(edges, n, asym=False):
    # Add Self-Loops
    edges = torch.cat((
        torch.arange(X.shape[0]).view(1, -1).repeat(2, 1), edges), dim=1)
    deg = torch.zeros(n).to(edges.device)
    deg.scatter_add_(
        dim=0, index=edges[0],
        src=torch.ones(edges.shape[1]).to(edges.device))
    # with open('degree_counts/%s_train.txt' % args.dataset, 'w') as file:
    #     for xs in deg.unique(sorted=True, return_counts=True):
    #         file.write(','.join('%d' % x for x in xs))
    #         file.write('\n')
    if asym:
        val = (deg ** -1)[edges[0]]
    else:
        val = (deg ** -0.5)[edges].prod(dim=0)
    return coo(edges, val, (n, n))


ev = Stat()

# Preprocessing
ev.start_preprocessing()

X, Y = map(gpu, [X, Y])
if args.method in ('SGC', 'IGNN') or args.correct or args.smooth:
    A = norm_adj(E, n_nodes).to(X.device)
    if args.method == 'SGC':
        for _ in range(args.n_layers):
            X = A @ X
        args.n_layers = 3
E = gpu(E)

ev.stop_preprocessing()

for run in range(args.runs):
    train_mask = train_masks[run]
    valid_mask = valid_masks[run]
    test_mask = test_masks[run]
    train_y = Y[train_mask].float()
    if not is_multiclass:
        train_y = F.one_hot(Y[train_mask], n_labels).float()
    if args.inductive:
        induc_mask = gpu(valid_mask | test_mask)[E].sum(dim=0).bool()
        induc_E = E[:, ~induc_mask]
    else:
        induc_E = E

    torch.manual_seed(run)
    ev.start_run()

    prs = {}
    wocs = copy.deepcopy(args)
    wocs.runs = 0
    wocs.correct = wocs.correct_rate = wocs.smooth = wocs.smooth_rate = 0
    wocs = str(wocs)
    prs_fn = 'predictions/%s-%s-%s.json' % (
        args.method,
        args.dataset,
        hashlib.md5(wocs.encode()).hexdigest(),
    )
    if os.path.exists(prs_fn):
        try:
            with open(prs_fn) as file:
                prs = json.load(file)
        except Exception:
            print('Load predictions failed:', prs_fn)
            prs = {}

    if (args.correct or args.smooth) and (
            prs.get('settings') == wocs and ('run-%d' % run in prs)):
        ev.best_acc, ev.best_y = prs['run-%d' % run]
        ev.best_y = gpu(torch.tensor(ev.best_y))
        print('Load predictions made at: %s' % prs['date'])
        opt = None
    elif args.method == 'IGNN':
        # https://arxiv.org/abs/2009.06211
        if args.dataset.startswith('chain-'):
            from models_chains import IGNN
        else:
            from models_heterophilic import IGNN
        if args.inductive:
            induc_A = gpu(norm_adj(induc_E.cpu(), n_nodes))
        else:
            induc_A = A
        net = IGNN(
            n_features, args.hidden, n_labels, n_nodes, args.dropout
        ).to(X.device)
        opt = Optim([*net.parameters()])
        x = X.T
        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                z = net(x, induc_A)
                criterion(z[train_mask], Y[train_mask]).mean().backward()
            ev.record_training(opt.elapsed)
            if args.inductive:
                z = net(x, A)
            if ev.evaluate_result(sg(z)):
                break
    elif args.method == 'EIGNN':
        # https://arxiv.org/abs/2202.10720
        # https://github.com/liu-jc/EIGNN
        if args.dataset.startswith('chain-'):
            from models_chains import EIGNN_Linear as EIGNN
        else:
            from models_heterophilic import IDM_SGC_Linear as EIGNN
        if run == 0:
            from normalization import aug_normalized_adjacency
            from utils import sparse_mx_to_torch_sparse_tensor
            import scipy.sparse as sp
            adj = sp.coo_matrix(
                (torch.ones(E.shape[1]), E.cpu()),
                shape=(n_nodes, n_nodes))
            sp_adj = aug_normalized_adjacency(adj)
            adj = sparse_mx_to_torch_sparse_tensor(sp_adj, device=X.device)
        net = EIGNN(
            adj, sp_adj, n_features, n_labels, num_eigenvec=100, gamma=0.8
        ).to(X.device)
        if args.dataset.startswith('chain-'):
            net.EIGNN.Lambda_S = net.EIGNN.Lambda_S.to(X.device)
            net.EIGNN.Q_S = net.EIGNN.Q_S.to(X.device)
        else:
            net.IDM_SGC.Lambda_S = net.IDM_SGC.Lambda_S.to(X.device)
            net.IDM_SGC.Q_S = net.IDM_SGC.Q_S.to(X.device)
        opt = Optim([*net.parameters()])
        x = X.T
        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                z = net(x)
                criterion(z[train_mask], Y[train_mask]).mean().backward()
            ev.record_training(opt.elapsed)
            if ev.evaluate_result(sg(z)):
                break
    else:
        # GNNs with Label (Label as Input & Label Reuse)
        # https://arxiv.org/abs/2103.13355

        in_feats, x = n_features, X
        if args.input_label:
            in_feats += n_labels
            state = torch.zeros((n_nodes, n_labels)).to(X.device)
            fwd_contracts = []
            bkd_contracts = []

        if args.method in ('MLP', 'SGC'):
            net = gpu(gnn.MLP(
                [in_feats, *([args.hidden] * (args.n_layers - 1)), n_labels],
                dropout=args.dropout))
            fwd = net.forward
            net.forward = lambda x, E: fwd(x)
        else:
            if args.method == 'JKNet':
                net = JKNet(in_feats, n_labels, **args.__dict__)
            elif args.method == 'GCNII':
                net = GCNII(in_feats, n_labels, **args.__dict__)
            elif args.method == 'GAT':
                net = gnn.GAT(
                    in_feats, args.hidden, args.n_layers, n_labels,
                    args.dropout, heads=args.heads)
            else:
                net = {
                    'GIN': gnn.GIN, 'GCN': gnn.GCN, 'SAGE': gnn.GraphSAGE
                }[args.method](
                    in_feats, args.hidden, args.n_layers,
                    n_labels, args.dropout)
            net = gpu(net)
        opt = Optim([*net.parameters()])

        for epoch in range(1, 1 + args.max_epochs):
            with opt:
                with torch.no_grad():
                    input_mask = sup_mask = train_mask
                    input_y = train_y
                    if args.input_label:
                        mask_label = 1 - args.input_label
                        if args.input_label > 1:
                            mask_label = 0.5
                            if fwd_contracts and fwd_contracts[-1]:
                                mask_label = max(0.1, min(
                                    0.9, fwd_contracts[-1] - exp))
                        mask = torch.rand(n_nodes) < mask_label
                        input_mask = train_mask & ~mask
                        sup_mask = train_mask & mask
                        input_y = train_y[input_mask[train_mask]]
                        # if not sup_mask.any().item():
                        #     sup_mask = train_mask
                        #     mask_label = 0
                        state[:] = 0
                        if args.for_iter:
                            net.eval()
                            last_dist = inf
                            fwd_contracts.append(0)
                            for _ in range(args.for_iter):
                                last_state = state.clone()
                                state[input_mask] = input_y
                                if args.back_iter:
                                    state[mask] = 0
                                    state /= (1 - mask_label)
                                state = sg(net(
                                    torch.cat((X, state), dim=1), induc_E))
                                # dist = (state - last_state).norm(2).item()
                                dist = norm(state - last_state).item()
                                if thres_for >= dist:
                                    break
                                fwd_contracts[-1] = max(
                                    dist / last_dist, fwd_contracts[-1])
                                last_dist = dist
                                # Stop iterating if the mapping is expanding
                                # NOTE: Only when using adaptive input_label to
                                # avoid impacting ablation studies
                                if (args.input_label > 1
                                        and fwd_contracts[-1] - 1 > exp):
                                    break
                            net.train()

                    # Masked Label Strategy for Backward-pass
                    if args.drop_state:
                        drop_state = args.drop_state
                        # Adaptively drop state wrt the contractive factor
                        if args.drop_state >= 1:
                            drop_state = 0.5
                            if bkd_contracts and bkd_contracts[-1]:
                                drop_state = max(0.1, min(
                                    0.9, bkd_contracts[-1] - exp))
                        mask = torch.rand(n_nodes) < drop_state
                        sup_mask = train_mask & mask
                        input_mask = train_mask & (~mask)
                        input_y = train_y[input_mask[train_mask]]
                        # if not sup_mask.any().item():
                        #     sup_mask = train_mask
                        #     drop_state = 0

                    if args.input_label:
                        state[input_mask] = input_y
                        if args.back_iter:
                            state[mask] = 0
                            # Back propagate through masked sigmoid/softmax
                            dsg = state * (1 - state)
                            # NOTE: rescale factor should not be taped
                            # because it simulates the sum of inputted
                            # and masked states feeding to the next layer
                            state /= (
                                (1 - drop_state) if args.drop_state
                                else (1 - mask_label))

                # Training
                if args.input_label:
                    if args.back_iter:
                        state.grad = None
                        state.requires_grad_(True)
                    x = torch.cat((X, state), dim=1)
                z = net(x, induc_E)
                if args.back_iter:
                    criterion(
                        z[sup_mask], Y[sup_mask]
                    ).mean().backward(retain_graph=True)
                    # Implicit Differentiation
                    last_vjpnorm = inf
                    bkd_contracts.append(0)
                    for _ in range(args.back_iter):
                        state.grad, vjp = None, state.grad
                        z.backward(vjp * dsg, retain_graph=True)
                        # vjpnorm = vjp.norm(2).item()
                        vjpnorm = norm(vjp).item()
                        if thres_back >= vjpnorm:
                            break
                        bkd_contracts[-1] = max(
                            vjpnorm / last_vjpnorm, bkd_contracts[-1])
                        # Stop iterating if the mapping is expanding
                        # NOTE: Only when using adaptive drop_state to
                        # avoid impacting ablation studies
                        if (args.drop_state >= 1
                                and bkd_contracts[-1] - 1 > exp):
                            break
                        last_vjpnorm = vjpnorm
                else:
                    criterion(z[sup_mask], Y[sup_mask]).mean().backward()
            ev.record_training(opt.elapsed)

            # Inference
            t = time.time()
            with torch.no_grad():
                net.eval()
                if args.input_label:
                    state[:] = 0
                    if args.for_iter:
                        last_dist = inf
                        for _ in range(args.for_iter):
                            last_state = state.clone()
                            state[train_mask] = train_y
                            state = sg(net(torch.cat((X, state), dim=1), E))
                            dist = norm(state - last_state).item()
                            if thres_for >= dist:
                                break
                            last_dist = dist
                    state[train_mask] = train_y
                    x = torch.cat((X, state), dim=1)
                state = sg(net(x, E))
                net.train()
            ev.record_evaluation(time.time() - t)
            if ev.evaluate_result(state):
                break

        if args.for_iter:
            print('forward contracts:', fwd_contracts)
            print('forward contract:',
                  fwd_contracts[-args.early_stop_epochs - 1])
        if args.back_iter:
            print('backward contracts:', bkd_contracts)
            print('backward contract:',
                  bkd_contracts[-args.early_stop_epochs - 1])

    try:
        with open(prs_fn, 'w') as file:
            prs['settings'] = wocs
            prs['date'] = str(datetime.datetime.now())
            prs['run-%d' % run] = [ev.best_acc, ev.best_y.cpu().tolist()]
            json.dump(prs, file)
    except Exception:
        pass

    # Correct and Smooth
    # https://arxiv.org/abs/2010.13993
    if args.correct:
        best_acc, best_y = ev.best_acc, ev.best_y
        y = best_y
        t = time.time()
        true_err = train_y - y[train_mask]
        err = torch.zeros(y.shape).to(y.device)
        for _ in range(args.correct):
            err[train_mask] = true_err
            err = (1 - args.correct_rate) * err + args.correct_rate * (A @ err)
            y = y + err
            acc = get_score(
                Y[valid_mask],
                y[valid_mask] > 0 if is_multiclass
                else y[valid_mask].argmax(dim=1))
            if acc > best_acc:
                best_acc, best_y = acc, y
        ev.record_training(time.time() - t)
        ev.evaluate_result(best_y)
    if args.smooth:
        best_acc, best_y = ev.best_acc, ev.best_y
        y = best_y
        t = time.time()
        for _ in range(args.smooth):
            y[train_mask] = train_y
            y = (1 - args.smooth_rate) * y + args.smooth_rate * (A @ y)
            acc = get_score(
                Y[valid_mask],
                y[valid_mask] > 0 if is_multiclass
                else y[valid_mask].argmax(dim=1))
            if acc > best_acc:
                best_acc, best_y = acc, y
        ev.record_training(time.time() - t)
        ev.evaluate_result(best_y)

    ev.end_run()
ev.end_all()
print('params: 0' if opt is None else opt)
print('script time:', time.time() - script_time)
