import argparse
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import seaborn as sns
from sklearn.metrics import confusion_matrix
from minepy import pstats, cstats
from scipy.stats import norm
import scipy.stats as stats




def get_common_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=98, help='seed')
    parser.add_argument('--server_tag', type=str, default='seizure', help='server_tag')
    parser.add_argument('--out_middle_features', action='store_true', help='out_middle_features')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    # dataset:
    parser.add_argument('--task', type=str, default='seizure', help='eeg task type')
    parser.add_argument('--dataset', type=str, default='SEED', help='SEED, SEED_IV')
    parser.add_argument('--data_path', type=str, default='./data/METR-LA', help='data path')
    parser.add_argument('--adj_file', type=str, default='./data/sensor_graph/adj_mx.pkl',
                        help='adj data path')
    parser.add_argument('--adj_type', type=str, default='scalap', help='adj type', choices=ADJ_CHOICES)

    # EEG specified:
    parser.add_argument('--testing', action='store_true', help='testing')
    parser.add_argument('--arg_file', type=str, default='None', help='chose saved arg file')
    parser.add_argument('--independent', action='store_true', help='subject independent')
    parser.add_argument('--using_fc', action='store_true', help='using_fc')

    parser.add_argument('--unit_test', action='store_true')
    parser.add_argument('--multi_train', action='store_true')

    parser.add_argument('--focalloss', action='store_true', help='focalloss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal_gamma')
    parser.add_argument('--weighted_ce', type=str, help='weighted cross entropy opt')
    parser.add_argument('--dev', action='store_true', help='dev')
    parser.add_argument('--dev_size', type=int, default=1000, help='dev_sample_size')
    parser.add_argument('--best_model_save_path', type=str, default='.best_model', help='best_model')
    parser.add_argument('--pre_model_path', type=str, default='./best_models/seed_pretrain_08021405', help='pre_model_path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.92, help='lr_decay_rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--clip', type=int, default=3, help='clip')
    parser.add_argument('--seq_length', type=int, default=3, help='seq_length')
    parser.add_argument('--predict_len', type=int, default=12, help='predict_len')
    parser.add_argument('--scheduler', action='store_true', help='scheduler')
    parser.add_argument('--mo', type=float, default=0.1, help='momentum')


    # running params
    parser.add_argument('--cuda', action='store_true', help='cuda')
    parser.add_argument('--transpose', action='store_true', help='transpose sequence and feature?')
    parser.add_argument('--runs', type=int, default=1, help='runs')
    parser.add_argument('--fig_filename', type=str, default='./mae', help='fig_filename')

    # choosing gnn model:
    parser.add_argument('--not_using_gnn', action='store_true', help='not_using_gnn')
    parser.add_argument('--gnn_name', type=str, default='gwn', help='gnn_name: gcn or gwn for now.')

    # GNN common params:
    parser.add_argument('--gnn_pooling', type=str, default='gate', help='gnn pooling')
    parser.add_argument('--agg_type', type=str, default='gate', help='gnn pooling')
    parser.add_argument('--gnn_layer_num', type=int, default=3, help='gnn_layer_num')
    parser.add_argument('--gnn_hid_dim', type=int, default=64, help='gnn_hid_dim')
    parser.add_argument('--gnn_out_dim', type=int, default=64, help='gnn_out_dim')
    parser.add_argument('--gnn_fin_fout', type=str, default='1100,550;550,128;128,128',
                        help='gnn_fin_fout for each layer')
    parser.add_argument('--gnn_res', action='store_true', help='gnn_res')
    parser.add_argument('--gnn_adj_type',  type=str, default='None', help='gnn_adj_type')
    parser.add_argument('--gnn_downsample_dim', type=int, default=0, help='gnn_downsample_dim')


    # gwn model params
    parser.add_argument('--coarsen_switch', type=int, default=3,
                        help='coarsen_switch: 0: sum, 1: gated, 2: avg, 3: concat.')
    parser.add_argument('--using_cnn', action='store_true', help='using_cnn')
    parser.add_argument('--gate_t', action='store_true', help='gate_t')
    parser.add_argument('--att', action='store_true', help='attention')
    parser.add_argument('--recur', action='store_true', help='recur')
    parser.add_argument('--fusion', action='store_true', help='fusion')
    parser.add_argument('--pretrain', action='store_true', help='pretrain')
    parser.add_argument('--feature_len', type=int, default=3, help='input feature_len')

    parser.add_argument('--gwn_out_features', type=int, default=32, help='gwn_out_features')
    parser.add_argument('--wavelets_num', type=int, default=20, help='wavelets_num')
    parser.add_argument('--rnn_layer_num', type=int, default=2, help='rnn_layer_num')
    parser.add_argument('--rnn_in_channel', type=int, default=32, help='rnn_in_channel')

    parser.add_argument('--rnn', action='store_true', help='attention')
    parser.add_argument('--bidirect', action='store_true', help='bidirect')


    # gcn params 
    parser.add_argument('--gcn_out_features', type=int, default=32, help='gcn_out_features')
    parser.add_argument('--rnn_hidden_len', type=int, default=32, help='rnn_hidden_len')
    parser.add_argument('--max_diffusion_step', type=int, default=2, help='max_diffusion_step')

    # eeg params
    parser.add_argument('--eeg_seq_len', type=int, default=250, help='eeg_seq_len')
    parser.add_argument('--predict_class_num', type=int, default=4, help='predict_class_num')

    # NOTE: encoder param:gnn_res
    parser.add_argument('--encoder', type=str, default='gnn', help='encoder')
    parser.add_argument('--encoder_hid_dim', type=int, default=256, help='encoder_out_dim')

    # NOTE: decoder param:
    parser.add_argument('--cut_encoder_dim', type=int, default=-1, help='cut_encoder_dim')
    parser.add_argument('--decoder', type=str, default='gnn', help='decoder')
    parser.add_argument('--decoder_type', type=str, default='conv2d', help='decoder_type')
    parser.add_argument('--decoder_downsample', type=int, default=-1, help='decoder_downsample')
    parser.add_argument('--decoder_hid_dim', type=int, default=512, help='decoder_hid_dim')
    parser.add_argument('--decoder_out_dim', type=int, default=32, help='decoder_out_dim')
    
    parser.add_argument('--predictor_num', type=int, default=3, help='predictor_num')
    parser.add_argument('--predictor_hid_dim', type=int, default=512, help='predictor_hid_dim')

    #NOTE: LatentGraphGenerator:
    parser.add_argument('--em_train', action='store_true', help='em_train, alternatively update grad')
    parser.add_argument('--lgg', action='store_true', help='lgg')
    parser.add_argument('--lgg_time', action='store_true', help='lgg time step')
    parser.add_argument('--lgg_warmup', type=int, default=10, help='lgg_warmup')
    parser.add_argument('--lgg_tau', type=float, default=0.01, help='gumbel softmax tau')
    parser.add_argument('--lgg_hid_dim', type=int, default=3, help='lgg_hid_dim')
    parser.add_argument('--lgg_k', type=int, default=3, help='lgg k component')


    # NOTE: DCRNN baseline:
    parser.add_argument('--dcgru_activation', type=str, default='tanh', help='dcgru_activation')

    return parser

def set_grad(m, requires_grad):
    for p in m.parameters():
        p.requires_grad = requires_grad

def freeze_module(m):
    set_grad(m, False)

def unfreeze_module(m):
    set_grad(m, True)

class DaoLogger:
    def __init__(self) -> None:
        self.debug_mode = False

    def init(self, args):
        self.args = args
        self.debug_mode = args.debug
    
    def log(self, *paras):
        print('[DLOG] ', *paras)
    
    def debug(self, *paras):
        if self.debug_mode: print('[DEBUG] ', *paras)

DLog = DaoLogger()

        
def normalize(data, fill_zeroes=True):
    '''
        only norm numpy type data with last dimension.
    '''
    mean = np.mean(data)
    std = np.std(data)
    if fill_zeroes:
        mask = (data == 0)
        data[mask] = mean
    return (data - mean) / std


class SeqDataLoader(object):
    def __init__(self, xs, ys, batch_size, cuda=False, pad_with_last_sample=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            # batch
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size) + 1
        if (self.num_batch - 1) * self.batch_size == self.size:
            self.num_batch -= 1

        print('num_batch ', self.num_batch)
        xs = torch.Tensor(xs)
        ys = torch.LongTensor(ys)
        if cuda:
            xs, ys = xs.cuda(), ys.cuda()
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            # TODO: Bug, we need to add more conditions.
            start_ind = 0
            end_ind = 0
            while self.current_ind < self.num_batch and start_ind <= end_ind and start_ind <= self.size:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()

def norm(tensor_data, dim=0):
    mu = tensor_data.mean(axis=dim, keepdim=True)
    std = tensor_data.std(axis=dim, keepdim=True)
    return (tensor_data - mu) / (std + 0.00005)


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    print('origin adj:', adj)
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def sym_norm_lap(adj):
    N = adj.shape[0]
    adj_norm = sym_adj(adj)
    L = np.eye(N) - adj_norm
    return L


def conv_L(in_len, kernel, stride, padding=0):
    ''' get the convolution output len
    '''
    return int((in_len - kernel + 2 * padding) / stride) + 1


def get_conv_out_len(in_len, modules):
    from torch import nn
    out_len = 0
    for i in range(len(modules)):
        m = modules[i]
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.AvgPool1d):
            out_len = conv_L(in_len, m.kernel_size[0], m.stride[0])
            in_len = out_len
            print('conv1d: in_len', in_len)
        elif isinstance(m, nn.MaxPool1d):
            out_len = conv_L(in_len, m.kernel_size, m.stride)
            in_len = out_len
            print('Pooling: in_len', in_len)
        elif isinstance(m, nn.Sequential):
            out_len = get_conv_out_len(in_len, m)
            in_len = out_len
        else:
            print("not counting!", m)

    return out_len


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt).toarray()
    normalized_laplacian = sp.eye(adj.shape[0]) - np.matmul(np.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


ADJ_CHOICES = ['laplacian', 'origin', 'scalap', 'normlap', 'symnadj', 'transition', 'identity','er']


def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    adj_mx[adj_mx > 0] = 1
    adj_mx[adj_mx < 0] = 0

    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32)]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "sym_norm_lap":
        adj = [sym_norm_lap(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

def calc_eeg_accuracy(preds, labels):
    """
    ACC, R, F1, MPR.
    """
    # return whole acc and each acc:
    num = preds.size(0)
    preds_b = preds.argmax(dim=1).squeeze()
    labels = labels.squeeze()
    ones = torch.zeros(num)
#     print(preds.shape, labels.shape)
    ones[preds_b == labels] = 1
    acc = torch.sum(ones) / num

    preds_dict = dict(Counter(preds_b))
    labels_dict = dict(Counter(labels))
    acc_each_dict = {}
    for k, v in labels_dict.items():
        acc_each_dict[k] = preds_dict[k]/v if k in preds_dict else 0
    

    return acc



def calc_metrics(preds, labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    # handle all zeros.
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    mse = (preds - labels) ** 2
    mae = torch.abs(preds - labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna(l, mask) for l in [mae, mape, mse]]
    rmse = torch.sqrt(mse)
    return mae, mape, rmse


def calc_metrics_eeg(preds, labels, criterion):
    labels = labels.squeeze()
    b = preds.shape[0]
    loss = criterion(preds, labels)
    return loss


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_tstep_metrics(model, test_loader, scaler, realy, seq_length) -> pd.DataFrame:
    model.eval()
    outputs = []
    for _, (x, __) in enumerate(test_loader.get_iterator()):
        testx = torch.Tensor(x).cuda().transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze(1))
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    test_met = []

    for i in range(seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        pred = torch.clamp(pred, min=0., max=70.)
        real = realy[:, :, i]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['mae', 'mape', 'rmse']).rename_axis('t')
    return test_met_df, yhat


def _to_ser(arr):
    return pd.DataFrame(arr.cpu().detach().numpy()).stack().rename_axis(['obs', 'sensor_id'])


def make_pred_df(realy, yhat, scaler, seq_length):
    df = pd.DataFrame(dict(y_last=_to_ser(realy[:, :, seq_length - 1]),
                           yhat_last=_to_ser(scaler.inverse_transform(yhat[:, :, seq_length - 1])),
                           y_3=_to_ser(realy[:, :, 2]),
                           yhat_3=_to_ser(scaler.inverse_transform(yhat[:, :, 2]))))
    return df


def correlation_map(data):
    data = np.array(data)
    d_shape = data.shape
    print('input data shape:', d_shape)  # (15, 45, 62, 300, 5)
    # 1. take subject - trial - channel, so the data will be: (N x seq)
    for sub in data:
        # take out one trial:
        t = sub[0]
        chan_len = d_shape[-1]
        print('total chan_len: ', chan_len)
        for i in range(d_shape[-1]):
            chan = t[..., i].transpose()

            # plot correaltion map
            pd_chan = pd.DataFrame(chan)
            print(pd_chan.corr())


def load_eeg_adj(adj_filename, adjtype=None):
    if 'npy' in adj_filename:
        adj = np.load(adj_filename)
    else:
        adj = np.genfromtxt(adj_filename, delimiter=',')
    adj_mx = np.asarray(adj)
    if adjtype in ["scalap", 'laplacian']:
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "sym_norm_lap":
        adj = [sym_norm_lap(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adjtype == "origin":
        return adj_mx

    return adj[0]