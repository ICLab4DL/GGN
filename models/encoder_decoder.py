import math
import os

import torch
import torch.nn.functional as F
from torch import  nn

import eeg_util
from eeg_util import DLog
from models.graph_conv_layer import *



def conv_L(in_len, kernel, stride, padding=0):
    return int((in_len - kernel + 2 * padding) / stride + 1)

def cal_cnn_outlen(modules, in_len, height=True):
    conv_l = in_len
    pos = 0 if height else 1
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[0], m.padding[pos])
            in_len = conv_l
        if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
            conv_l = conv_L(in_len, m.kernel_size[pos], m.stride, m.padding)
            in_len = conv_l                
    return conv_l

class CNN2d(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, width, height, kernel=(3, 3), stride=1, layers=2, dropout=0.6, pooling=False):
        super(CNN2d, self).__init__()
        self.dropout = dropout
        self.pooling = pooling
        b1_dim = int(hid_dim/2)

        self.b1 = self.cnn_block(in_dim, b1_dim)
        if pooling:
            self.pool1 = nn.MaxPool2d((3,3), 2)

        self.bx = nn.ModuleList()
        for _ in range(layers-2):
            self.bx.append(self.cnn_block(b1_dim, hid_dim, kernel, stride))

        self.bn = self.cnn_block(hid_dim, b1_dim, kernel, stride)

        if pooling:
            self.pool2 = nn.AvgPool2d((2,2), 3)
            

        self.len_h = cal_cnn_outlen(self.modules(), height)
        self.len_w = cal_cnn_outlen(self.modules(), width, False)

        DLog.log('CNNEncoder2d out len:', self.len_h, self.len_w)
        
        self.l1 = nn.Linear(b1_dim * self.len_h * self.len_w, out_dim)
        
        DLog.log('CNNEncoder2d struct:', self)
        
    def cnn_block(self, in_dim, out_dim, kernel=(3, 3), stride=1):
        block = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(1,1)),
                  nn.ReLU(),
                  nn.Conv2d(out_dim, out_dim, kernel_size=kernel, stride=stride),
                  nn.ReLU(),
                  nn.BatchNorm2d(out_dim),
                  nn.Dropout(self.dropout))
        return block

    def forward(self, x):
        """ 
        input x: (B C, N, T)
        outout x:
        """
        DLog.debug('conv2d in', x.shape)
        x = self.b1(x)
        if self.pooling:
            x = self.pool1(x)
        for b in self.bx:
            x = b(x)
        x = self.bn(x)
        if self.pooling:
            x = self.pool2(x)

        x = self.l1(torch.flatten(x, start_dim=1))
        return x

    def reset_parameters(self):
        pass

class CNN1d(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, height, kernel=3, 
                    stride=1, layers=2, dropout=0.6,tag=None, linear=False):
        super(CNN1d, self).__init__()
        self.dropout = dropout
        self.linear = linear
        b1_dim = int(hid_dim/2)

        self.b1 = self.cnn_block(in_dim, b1_dim)
        self.bx = nn.ModuleList()
        for _ in range(layers-2):
            self.bx.append(self.cnn_block(b1_dim, hid_dim, kernel, 2))

        self.bn = self.cnn_block(hid_dim, out_dim, kernel, stride)

        self.width_len = cal_cnn_outlen(self.modules(), height)

        DLog.log(f'{tag}: CNNEncoder out len:', self.width_len)
        
        if self.linear:
            self.l1 = nn.Linear(out_dim * self.width_len, out_dim)
        
    def cnn_block(self, in_dim, out_dim, kernel=3, stride=1):
        block = nn.Sequential(nn.Conv1d(in_dim, out_dim, kernel_size=1),
                  nn.ReLU(),
                  nn.Conv1d(out_dim, out_dim, kernel_size=kernel, stride=stride),
                  nn.ReLU(),
                  nn.BatchNorm1d(out_dim),
                  nn.Dropout(self.dropout))
        return block


    def forward(self, x):
        """
        input x: (B, C, T) : [32 * 20 or 32 * 1, 244, 34]
        outout x:
        """
        x = self.b1(x)
        for b in self.bx:
            x = b(x)
        x = self.bn(x)
        if self.linear:
            x = self.l1(torch.flatten(x, start_dim=1))

        return x

class LSTMEncoder(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, bidirect=False):
        super(LSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=in_dim,
                          hidden_size=hid_dim,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=bidirect)
    
    def forward(self, x):
        B,C,N,T = x.shape
        x = x.transpose(1, 2).reshape(B*N, C, T).transpose(1, 2)
        x, h = self.rnn(x, None)
        x = x.reshape(B, N, T, -1).transpose(2, 3)
        return x
    
    
class RNNEncoder(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim, bidirect=False):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(input_size=in_dim,
                          hidden_size=hid_dim,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=bidirect)
    
    def forward(self, x):
        B,C,N,T = x.shape
        x = x.transpose(1, 2).reshape(B*N, C, T).transpose(1, 2)
        x, h = self.rnn(x, None)
        x = x.reshape(B, N, T, -1).transpose(2, 3)
        return x
        
    

class MultiEncoders(nn.Module):
    def __init__(self, args, in_dim, hid_dim, out_dim):
        super(MultiEncoders, self).__init__()
        self.encoder1 = MultiCNNEncoder(1, args, in_dim, hid_dim, out_dim,
                                        height=34, kernel=3, stride=1, layers=3, tag="Encoder1", linear=True)
        self.encoder2 = MultiCNNEncoder(1, args, in_dim, hid_dim, out_dim,
                                        height=34, kernel=5, stride=2, layers=3,tag="Encoder2",linear=True)
        self.width_len = self.encoder1.width_len + self.encoder2.width_len
        
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x)
        return torch.cat([x1, x2], dim=-1)
    

class MultiCNNEncoder(nn.Module):
    def __init__(self, cnn_num, args, in_dim, hid_dim, out_dim, height, kernel=3, 
                 stride=1, layers=2,tag=None, linear=False):
        super(MultiCNNEncoder, self).__init__()
        self.cnn_num = cnn_num
        self.cnns = nn.ModuleList()
        self.shared = True
        self.linear = linear
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        for _ in range(self.cnn_num):
            self.cnns.append(CNN1d(in_dim, hid_dim, out_dim, height, kernel,
                                          stride, layers, args.dropout,tag=tag, linear=linear))
        self.width_len = self.cnns[-1].width_len
        
    def forward(self, x):
        B,C,N,T = x.shape
        DLog.debug('mul in shape:', x.shape)
        node_emb = []
        
        if self.shared:
            x = x.transpose(1, 2).reshape(B*N, C, T)
            for i, cnn in enumerate(self.cnns):
                node_emb_tmp = []
                node_emb_tmp = cnn(x)
                # for j in range(N):
                    # node_emb_tmp.append(cnn(x[:,:,j,:]))  # outshape: B, C1
                # node_emb.append(torch.stack(node_emb_tmp, dim=1)) # B,N,C1
                node_emb.append(node_emb_tmp)
            node_embs = torch.stack(node_emb, dim=0).max(dim=0).values
            if self.linear:
                node_embs = node_embs.reshape(B, N, -1)
            else:
                node_embs = node_embs.reshape(B, N, self.out_dim, -1)
        else:
            assert N == len(self.cnns)
            for i, cnn in enumerate(self.cnns):
                node_emb.append(cnn(x[:,:,i,:]))  # outshape: B, C1
            node_embs = torch.stack(node_emb, dim=1)  # B,N,C1, T
            
        return node_embs
        

class GNNDecoder(nn.Module):
    def __init__(self, N, args, in_dim, out_dim):
        super(GNNDecoder, self).__init__()
        self.gnns = nn.ModuleList()
        self.args = args
        self.N = N
        if args.gnn_downsample_dim > 0:
            self.downsample = nn.Linear(in_dim, args.gnn_downsample_dim)
            self.gnn_in_dim = args.gnn_downsample_dim
        else:
            self.gnn_in_dim = in_dim
            self.downsample = None
            
        self.gnns.append(GraphConv(N, self.gnn_in_dim, args.gnn_hid_dim, args.dropout))
        for _ in range(args.gnn_layer_num-2):
            self.gnns.append(GraphConv(N, args.gnn_hid_dim, args.gnn_hid_dim, args.dropout))
        self.gnns.append(GraphConv(N, args.gnn_hid_dim, out_dim, args.dropout))

        self.pooling = args.gnn_pooling not in ['None','0',0,'none']
        if args.gnn_pooling == 'att':
            self.g_pooling = AttGraphPooling(args, N, out_dim, out_dim)
        elif args.gnn_pooling == 'cpool':
            K = 3
            self.g_pooling = CompactPooling(args, K, N)
        elif args.gnn_pooling == 'cat':
            self.g_pooling = CatPooling()
        else:
            self.g_pooling = GateGraphPooling(args, N)
        
        self.adj_w = nn.Parameter(torch.Tensor(N, N).cuda())  # suppose gumbel distritbion? or Bern distritbution.

        self.reset_parameters()
        # sum (3xN - N x F )
        # gated
        # compare them

    def forward(self, adj, x):
        # x = self.gnn(adj, x)
        # x shape: B*NC
        B = x.shape[0]

        if self.downsample is not None:
            x = self.downsample(x)

        if adj is None:
            adj = self.adj_w
        origin_x = x
        for gnn in self.gnns:
            x = gnn(x, adj)
        
        if self.pooling:
            x = self.g_pooling(x)
            x = x.reshape(B, -1)

        if self.args.gnn_res:
            x = torch.cat([x, origin_x], dim=2)
        DLog.debug('gnn decoder out shape:', x.shape)
        return x

    def reset_parameters(self):
        # self.gnn.reset_parameters()
        nn.init.kaiming_normal_(self.adj_w)
        

class SpatialDecoder(nn.Module):
    def __init__(self, args, gnn_decoder=None, cnn_decoder=None):
        super(SpatialDecoder, self).__init__()
        self.args = args
        self.gnn_decoder = gnn_decoder
        self.cnn_decoder = cnn_decoder

    def forward(self, adj, x):
        # ([32, 34, 20, 64]) B T N C  # input.
        if isinstance(adj, list):
            x1 = []
            for t in range(len(adj)):
                xt = self.gnn_decoder(adj[t], x[:,t, :, :])
                x1.append(xt)
            x1 = torch.stack(x1, dim=1)
            x1 = torch.flatten(x1, start_dim=1)
        else:
            x1 = self.gnn_decoder(adj, x[:,-1, :, :]) # only take last hidden
        # to BCNT
        DLog.debug('DecoderAdapter x1 shape:', x1.shape)
        if self.cnn_decoder is not None:
            x = x.transpose(3, 1)
            x2 = self.cnn_decoder(x)
            DLog.debug('DecoderAdapter x2 shape:', x2.shape)
            # try concate
            x = torch.cat([x1, x2], dim=1)
        DLog.debug('SpatialDecoder out shape:', x.shape)
        return x