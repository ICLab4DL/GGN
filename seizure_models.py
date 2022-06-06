import math
import os

import torch
import torch.nn.functional as F
from torch import  nn

import eeg_util
from gnn_models import *
from eeg_util import DLog

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class FocalLoss(nn.Module):
    def __init__(self, celoss=None, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.celoss = celoss
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        if self.celoss is None:
            loss = F.cross_entropy(inputs,  targets, reduce=False)
        else:
            loss = self.celoss(inputs,  targets)
        p = torch.exp(-loss)
        flloss = torch.mean(self.alpha * torch.pow((1-p), self.gamma) * loss)
        return flloss


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

class CNNEncoder2d(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, width, height, kernel=(3, 3), stride=1, layers=2, dropout=0.6, pooling=False):
        super(CNNEncoder2d, self).__init__()
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

class CNNEncoder1d(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, height, kernel=3, 
                    stride=1, layers=2, dropout=0.6,tag=None, linear=False):
        super(CNNEncoder1d, self).__init__()
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

class CNNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.6):
        super(CNNEncoder, self).__init__()

        b1_dim = int(hid_dim/2)
        self.b1 = nn.Sequential(nn.Conv1d(in_dim, hid_dim, kernel_size=3, stride=2),
                  nn.ReLU(),
                  nn.Conv1d(hid_dim, b1_dim, kernel_size=1),
                  nn.ReLU(),
                  nn.BatchNorm1d(b1_dim),
                  nn.Dropout(dropout))

        self.b2 = nn.Sequential(nn.Conv1d(b1_dim, b1_dim, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.Conv1d(b1_dim, b1_dim, kernel_size=1),
                  nn.ReLU(),
                  nn.BatchNorm1d(b1_dim),
                  nn.Dropout(dropout))
        b2_dim = int(b1_dim/2)
        self.b3 = nn.Sequential(nn.Conv1d(b1_dim, b2_dim, kernel_size=3, stride=1),
                  nn.ReLU(),
                  nn.Conv1d(b2_dim, b2_dim, kernel_size=1),
                  nn.ReLU(),
                  nn.BatchNorm1d(b2_dim),
                  nn.Dropout(dropout))

        self.cnn_out_len = self.cal_cnn_outlen(34)
        self.l1 = nn.Linear(b2_dim*self.cnn_out_len, out_dim)
        DLog.log('CNNEncoder out len:', self.cnn_out_len)
        

    def cal_cnn_outlen(self, in_len):
        conv_l = in_len
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_l = conv_L(in_len, m.kernel_size[0], m.stride[0], m.padding[0])
                DLog.log('conv len:', conv_l)
                in_len = conv_l
        return conv_l

    def forward(self, x):
        """
        input x: (B * N, C, T) : [32 * 20, 244, 34]
        outout x:
        """
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.l1(torch.flatten(x, start_dim=1))
        return x

    def reset_parameters(self):
        self.b1.reset_parameters()
        self.b2.reset_parameters()
        self.b3.reset_parameters()
        self.l1.reset_parameters()


class GraphGenerator(nn.Module):
    def __init__(self, N, dim_in, args):
        super(GraphGenerator, self).__init__()
        self.N = N
        # self.W1 = nn.Parameter(torch.rand(dim_in, args.encoder_hid_dim))
        # self.W2 = nn.Parameter(torch.rand(args.encoder_hid_dim, 1))
        
        self.adj_w = nn.Parameter(torch.rand(N, N))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.adj_w)
        
    def foward(self, z):
        """
        input z: N x F
        output adj_z: N x N
        """
        # N, C = z.shape
        # zz = z.repeat(1, N) * (z.flatten())
        # zz = zz.reshape(N*N, C)
        # q = F.relu(torch.mm(zz,self.W1))
        # q = F.dropout(q, p=0.3)
        # q = torch.sigmoid(q.mm(self.W2).reshape(N, N))
        
        return self.adj_w


class GraphConv(nn.Module):
    def __init__(self, N, in_dim, out_dim, dropout):
        super(GraphConv, self).__init__()
        self.N = N
        self.dropout = dropout
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        # self.adj_w = nn.Parameter(torch.rand(N, N))
        self.reset_parameters()
        
    def forward(self, x, adj=None):
        # x shape is B*NC
        if adj is None:
            adj = self.adj_w
        DLog.debug('gnn x input shape', x.shape)
        DLog.debug('gnn adj shape', adj.shape)
        x = torch.matmul(adj, x)
        DLog.debug('gnn x output shape', x.shape)
        x = self.dropout(F.relu(self.lin(x)))
        return x

    def reset_parameters(self):
        pass
        # nn.init.kaiming_normal_(self.adj_w)
        
class MultilayerGNN(nn.Module):
    def __init__(self, N, layer_num, pooling, in_dim, hid_dim, out_dim, dropout=0.5):
        super(MultilayerGNN, self).__init__()
        self.gnns = nn.ModuleList()

        self.gnns.append(GraphConv(N, in_dim, hid_dim, dropout))
        for _ in range(layer_num-2):
            self.gnns.append(GraphConv(N, hid_dim, hid_dim, dropout))
        self.gnns.append(GraphConv(N, hid_dim, out_dim, dropout))

        # self.g_pooling = pooling
        
    def forward(self, x, adj):
        """
        input x shape: B*NC
        """
        # B = x.shape[0]
        for gnn in self.gnns:
            x = gnn(x, adj)
        
        # if self.g_pooling is not None:
        #     x = self.g_pooling(x)
        #     x = x.reshape(B, -1)
        DLog.debug('MultilayerGNN out shape:', x.shape)
        
        return x


class GNNDecoder(nn.Module):
    def __init__(self, N, args, in_dim, out_dim):
        super(GNNDecoder, self).__init__()
        # self.gnn = GraphSAGE(in_dim, args.gnn_hid_dim, args.decoder_out_dim, args.gnn_layer_num, args.dropout)
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
        

class ClassPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, class_num, num_layers,
                 dropout=0.5):
        super(ClassPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        DLog.log('Predictor in channel:', in_channels)
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, class_num))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        DLog.debug('input prediction x shape:', x.shape)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class CompactPooling(nn.Module):
    def __init__(self, args, K, N):
        super(CompactPooling, self).__init__()
        self.CompM = nn.Parameter(torch.Tensor(K, N).cuda())
        nn.init.normal_(self.CompM, mean=0.01, std=0.01)

    def forward(self, x):
        DLog.debug('CompactPooling in shape:', x.shape)
        x = torch.matmul(self.CompM, x)
        DLog.debug('matmul CompactPooling shape:', x.shape)
        x = torch.sum(x, dim=-2).squeeze()
        DLog.debug('out CompactPooling shape:', x.shape)
        return x

class GateGraphPooling(nn.Module):
    def __init__(self, args, N):
        super(GateGraphPooling, self).__init__()
        self.args = args
        self.N = N
        self.gate =nn.Parameter(torch.FloatTensor(self.N))
        self.reset_parameters()
        
        
    def forward(self, x):
        """ignore the following dimensions after the 3rd one.
        Args:
            x (tensor): shape: B,N,C,...
        Returns:
            x shape: B,C,...
        """
        shape = x.shape
        if len(shape) > 3:
            x = torch.einsum('btnc, n -> btc', x, self.gate)
        else:
            x = torch.einsum('bnc, n -> bc', x, self.gate)
        return x
    
    def reset_parameters(self):
        nn.init.normal_(self.gate, mean=0.01, std=0.01)
    

class CatPooling(nn.Module):
    def __init__(self):
        super(CatPooling, self).__init__()
        pass

    def forward(self, x):
        # input B*NC -> B*C
        return torch.flatten(x, start_dim=-2)
    

class AttGraphPooling(nn.Module):
    def __init__(self, args, N, in_dim, hid_dim):
        super(AttGraphPooling, self).__init__()
        self.args = args
        self.N = N
        self.Q = nn.Linear(in_dim, hid_dim)
        self.K = nn.Linear(in_dim, hid_dim)
        self.V = nn.Linear(in_dim, hid_dim)
        if args.agg_type == 'gate':
            self.gate =nn.Parameter(torch.FloatTensor(self.N))
        self.reset_parameters()
        
    def forward(self, x):
        """ignore the following dimensions after the 3rd one.
        Args:
            x (tensor): shape: B,N,C
        Returns:
            x shape: B,C
        """
        x = x.transpose(2, -1)  # make last dimension is channel.
        Q = self.Q(x) # BNC BNC.
        K = self.K(x)
        V = self.V(x)
        
        att = torch.bmm(Q, K.transpose(1,2))/self.N**0.5
        att = torch.softmax(att, dim=1)
        DLog.debug('att shape', att.shape)
        x = torch.bmm(att, V)  # bnc.
        # TODO: add gated? or sum? or linear?, try 3.
        if self.args.agg_type == 'gate':
        # NOTE: method1: gated
            x = torch.einsum('bnc, n -> bc', x, self.gate)
        # NOTE: method2:cat
        elif self.args.agg_type == 'cat':
        # NOTE: method3:sum
            x = torch.flatten(x, start_dim=1)
        elif self.args.agg_type == 'sum':
            x = torch.sum(x, dim=1)
        
        DLog.debug('after att:', x.shape)
        return x
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.Q.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.K.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.V.weight, mode='fan_in')
        if self.args.agg_type == 'gate':
            nn.init.normal_(self.gate, mean=0.01, std=0.01)
        

class FCDecoder(nn.Module):
    def __init__(self, args, in_dim, N, out_dim):
        super(FCDecoder, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(in_dim * N, out_dim)
                # nn.ReLU(),
                # nn.Dropout(args.dropout),
                # nn.Linear(512, out_dim),
                # nn.ReLU(),
                # nn.Dropout(args.dropout)
            )
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class GNNFCDecoder(nn.Module):
    def __init__(self, args, gnn_decoder, fc_decoder):
        super(GNNFCDecoder, self).__init__()
        self.args = args
        self.gnn_decoder = gnn_decoder
        self.fc_decoder = fc_decoder

    def forward(self, adj, x):
        x1 = self.gnn_decoder(adj, x)
        x2 = self.fc_decoder(x)
        DLog.debug('GNNFCDecoder x1 shape:', x1.shape)
        DLog.debug('GNNFCDecoder x2 shape:', x2.shape)
        # concat:
        x = torch.cat([x1, x2], dim=1)
        DLog.debug('GNNFCDecoder x shape:', x.shape)

        return x

class DecoderAdapter(nn.Module):
    def __init__(self, args, gnn_decoder=None, cnn_decoder=None):
        super(DecoderAdapter, self).__init__()
        self.args = args
        self.gnn_decoder = gnn_decoder
        self.cnn_decoder = cnn_decoder

    def forward(self, adj, x):
        if isinstance(adj, list):
            x1 = []
            for t in range(len(adj)):
                xt = self.gnn_decoder(adj[t], x[:,t, :, :])
                x1.append(xt)
            x1 = torch.stack(x1, dim=1)
            x1 = torch.flatten(x1, start_dim=1)
        else:
            x1 = self.gnn_decoder(adj, x[:,-1, :, :]) # only take last hidden
        # ([32, 34, 20, 64]) B T N C  # input.
        # to BCNT
        DLog.debug('DecoderAdapter x1 shape:', x1.shape)
        x = x.transpose(3, 1)
        x2 = self.cnn_decoder(x)
        DLog.debug('DecoderAdapter x2 shape:', x2.shape)
        # try concate
        x = torch.cat([x1, x2], dim=1)
        DLog.debug('DecoderAdapter out shape:', x.shape)
        return x



class TestDecoder(nn.Module):
    def __init__(self, args, N, in_dim, out_dim, width_len=13):
        super(TestDecoder, self).__init__()
        self.args = args
        self.N = N
        self.in_dim = in_dim
        self.gate = nn.Parameter(torch.Tensor(self.N))
        if args.decoder_type=='fc':
            self.test_decoder = FCDecoder(args, in_dim, N, out_dim)
        else:
            self.cnns = CNNEncoder2d(in_dim, args.decoder_hid_dim, out_dim, width=width_len, height=self.N, stride=2, layers=3, dropout=args.dropout)

    def forward(self, adj, x):
        B, T, N, C = x.shape
        DLog.debug('test decoder in shape:', x.shape)
        x = x.permute(0, 3, 2, 1) # BCNT

        # 1. test concate  # bad
        # return x.reshape(B, -1) 
        # 2. test gated.
        # x = torch.einsum('bnc, n -> bc', x, self.gate)
        # x = x / self.N
        # 3. 1d cnn for node-wise.
        
        # 4. NOTE pooling. topK pooling
        # x = torch.topk(x, 5, dim=1)
        # NOTE: 5. cnn decoder
        if self.args.decoder_type=='fc':
            x = self.test_decoder(x)
        else:
            x = self.cnns(x)
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
            self.cnns.append(CNNEncoder1d(in_dim, hid_dim, out_dim, height, kernel,
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
        
class GGN(nn.Module):
    def __init__(self, adj, args, out_mid_features=False):
        super(GGN, self).__init__()
        self.args = args
        self.log = False
        self.adj_eps = 0.1
        self.adj = adj
        self.adj_x = adj
        self.adj_0 = self.adj_to_coo_longTensor(adj)

        self.N = adj.shape[0]
        print('N:', self.N)
        en_hid_dim = args.encoder_hid_dim
        en_out_dim = 16
        self.out_mid_features = out_mid_features
        
        
        if args.encoder == 'rnn':
            self.encoder = RNNEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2

            de_out_dim = args.decoder_out_dim
            encoder_out_width = 34
        elif args.encoder == 'lstm':
            self.encoder = LSTMEncoder(args, args.feature_len, en_hid_dim, en_out_dim, args.bidirect)
            decoder_in_dim = en_hid_dim
            if args.bidirect:
                decoder_in_dim *= 2
            de_out_dim = args.decoder_out_dim
            encoder_out_width = 34
            
        elif args.encoder == 'cnn2d':
            cnn = CNNEncoder2d(in_dim=args.feature_len, 
                               hid_dim=en_hid_dim, 
                               out_dim=args.decoder_out_dim, 
                               width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.encoder = cnn
            de_out_dim = args.decoder_out_dim
        else:
            self.encoder = MultiEncoders(args, args.feature_len, en_hid_dim, en_out_dim)
            encoder_out_width = self.encoder.width_len
            decoder_in_dim = en_out_dim * 2
            de_out_dim = args.decoder_out_dim + decoder_in_dim

        if args.gnn_adj_type == 'rand':
            self.adj = None
            self.adj_0 = None

        if args.lgg:
            self.LGG = LatentGraphGenerator(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim,
                                        args.lgg_k)


        if args.decoder == 'gnn':
            if args.cut_encoder_dim > 0:
                decoder_in_dim *= args.cut_encoder_dim
            self.decoder = GNNDecoder(self.N, args, decoder_in_dim, de_out_dim)
            if args.agg_type == 'cat':
                de_out_dim *= self.N
        elif args.decoder == 'gnn_fc':
            if args.cut_encoder_dim > 0:
                decoder_in_dim *= args.cut_encoder_dim
            print('decode_out:', de_out_dim)
            gnn = GNNDecoder(self.N, args, decoder_in_dim, de_out_dim)
            fc = FCDecoder(args, decoder_in_dim, self.N, de_out_dim)
            self.decoder = GNNFCDecoder(args, gnn, fc)
            de_out_dim *= 2
        elif args.decoder == 'gnn_cnn':
            gnn = GNNDecoder(self.N, args, decoder_in_dim, args.gnn_out_dim)
            cnn_in_dim = decoder_in_dim
            cnn = CNNEncoder2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                            width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = DecoderAdapter(args, gnn, cnn)
            if args.agg_type == 'cat':
                de_out_dim += args.gnn_out_dim * self.N
            else:
                if args.lgg and args.lgg_time:
                    de_out_dim += args.gnn_out_dim * 34
                else:
                    de_out_dim += args.gnn_out_dim
        elif args.decoder == 'cnn2d':
            cnn = CNNEncoder2d(decoder_in_dim, args.decoder_hid_dim, de_out_dim, 
                            width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = DecoderAdapter(args, None, cnn)
        elif args.decoder.upper() == 'NONE':
            self.decoder = None
            de_out_dim = decoder_in_dim * self.N
        else:
            self.decoder = TestDecoder(args, self.N, decoder_in_dim, de_out_dim, width_len=encoder_out_width)
        self.graphG = GraphGenerator(self.N, en_out_dim, args)
        
        
        self.predictor = ClassPredictor(de_out_dim, hidden_channels=args.predictor_hid_dim,
                                class_num=args.predict_class_num, num_layers=args.predictor_num, dropout=args.dropout)

        self.warmup = args.lgg_warmup
        self.epoch = 0
        
        DLog.log('-------- ecoder: -----------\n', self.encoder)
        DLog.log('-------- decoder: -----------\n', self.decoder)
        
        self.reset_parameters()

    def adj_to_coo_longTensor(self, adj):
        """adj is cuda tensor
        """
        DLog.debug(adj)
        adj[adj > self.adj_eps] = 1
        adj[adj <= self.adj_eps] = 0

        idx = torch.nonzero(adj).T.long() # (row, col)
        DLog.debug('idx shape:', idx.shape)
        return idx

    def encode(self, x):
        # B,C,N,T = x.shape
        x = self.encoder(x)
        return x

    def fake_decoder(self, adj, x):
        DLog.debug('fake decoder in shape:', x.shape)
        # trans to BC:
        if len(x.shape) == 4:
            x = x[:,-1,...]
            
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        
        return x

    def decode(self, x, B, N, adj):

        if self.decoder is None:
            x = self.fake_decoder(adj, x)
            DLog.debug('decoder out shape:', x.shape)
            return x

        if self.args.cut_encoder_dim > 0:
            x = x[:,:,:,-self.args.cut_encoder_dim:]

        x = self.decoder(adj, x)
        
        DLog.debug('decoder out shape:', x.shape)
        return x

    def alternative_freeze_grad(self, epoch):
        self.epoch = epoch
        if self.epoch > self.warmup:
            if epoch % 2==0:
                # freeze LGG
                eeg_util.freeze_module(self.LGG)
                eeg_util.unfreeze_module(self.encoder)
            else:
                # freeze encoder
                eeg_util.freeze_module(self.encoder)
                eeg_util.unfreeze_module(self.LGG)
                

    def forward(self, x, *options):
        """
        input x shape: B, C, N, T
        output x: class
        """
        B,C,N,T = x.shape

        # (1) encoder:
        x = self.encode(x)

        # before: BNCT
        x = x.permute(0, 3, 1, 2)
        # permute to BTNC

        # (2) adj selection:

        # LGG, latent graph generator:
        if self.args.lgg:
            if self.args.lgg_time:
                adj_x_times = []
                for t in range(T):
                    x_t = x[:, t, ...]
                    if self.training:
                        if self.epoch < self.warmup:
                            adj_x = self.LGG(x_t, self.adj)
                        else:
                            adj_x = self.LGG(x_t)
                    else:
                        adj_x = self.LGG(x_t)
                        DLog.debug('Model is Eval!!!!!!!!!!!!!!!!!')
                    adj_x_times.append(adj_x)
                self.adj_x = adj_x_times
            else:
                x_t = x[:, -1, ...]  # NOTE: take last time step.
                if self.training and self.epoch < self.warmup:
                    self.adj_x = self.LGG(x_t, self.adj)
                else:
                    self.adj_x = self.LGG(x_t)
                    DLog.debug('Model is Eval!!!!!!!!!!!!!!!!!')

        # (3) decoder:
        DLog.debug('decoder input shape:', x.shape)
        
        x = self.decode(x, B, N, self.adj_x)
        DLog.debug('decoder output shape:', x.shape)

        if self.out_mid_features:
            return x
        
        # (4) predictor:
        x = self.predictor(x)
        return x

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.count=0



class Trainer:
    def __init__(self, args, model, optimizer=None, scaler=None, criterion=nn.MSELoss(), sched=None):
        self.model = model
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.clip = args.clip
        self.lr_decay_rate = args.lr_decay_rate
        self.epochs = args.epochs
        self.scheduler = sched

    def lr_schedule(self):
        self.scheduler.step()

    def train(self, input_data, target, epoch=-1):
        self.model.train()
        self.optimizer.zero_grad()

        # train
        output = self.model(input_data, 'train')
        if self.args.em_train:
            self.model.alternative_freeze_grad(epoch)

        output = output.squeeze()
        loss = eeg_util.calc_metrics_eeg(output, target, self.criterion)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), output.detach()

    def eval(self, input_data, target):
        self.model.eval()

        output = self.model(input_data)  # [batch_size,seq_length,num_nodes]
        output = output.squeeze()
        loss = eeg_util.calc_metrics_eeg(output, target, self.criterion)
        return loss.item(), output.detach()




class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.2)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        :param x: (batch, N, in_channel, Seq)
        :return: (batch, N, out_channel, Seq)
        """
        x = x.squeeze()
        B, N, C = x.shape
#         print('GCN input: ',x.shape)
        #         print('weigth shape:', self.weight.shape)
        x = torch.einsum("bnh, hf -> bnf", x, self.weight)
        x = torch.einsum("bnf, nm -> bmf", x, adj).reshape(B * N, -1)
        x = self.bn(x)
        x = self.dropout(x)
        #         support = torch.mm(x, self.weight)
        #         output = torch.spmm(adj, support)
        x = x.reshape(B, N, -1)
        
        if self.bias is not None:
            x = x + self.bias
        x = torch.sigmoid(x.unsqueeze(dim=-1))
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LatentGraphGenerator(nn.Module):
    def __init__(self, args, A_0, tau, in_dim, hid_dim, K=10):
        super(LatentGraphGenerator,self).__init__()
        self.N = A_0.shape[0] # num of nodes.
        self.tau = tau
        self.args = args
        self.A_0 = A_0
        self.args = args

        if args.gnn_pooling == 'att':
            pooling = AttGraphPooling(args, self.N, in_dim, 64)
        elif args.gnn_pooling == 'cpool':
            pooling = CompactPooling(args, 3, self.N)
        elif args.gnn_pooling.upper() == 'NONE':
            pooling = None
        else:
            pooling = GateGraphPooling(args, self.N)
            
        self.gumbel_tau = 0.1
        self.mu_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.sig_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        self.pi_nn = MultilayerGNN(self.N, 2, pooling, in_dim, hid_dim, K, args.dropout)
        
        self.adj_fix = nn.Parameter(self.A_0)

        print('adj_fix', self.adj_fix.shape)

        self.init_norm()


    def init_norm(self):
        self.Norm = torch.randn(size=(1000, self.args.batch_size, self.N)).cuda()
        self.norm_index = 0

    def get_norm_noise(self, size):
        if self.norm_index >= 999:
            self.init_norm()

        if size == self.args.batch_size:
            self.norm_index += 1
            return self.Norm[self.norm_index].squeeze()
        else:
            return torch.randn((size, self.N)).cuda()
        
    def update_A(self, mu, sig, pi):
        """ mu, sig, pi, shape: (B, N, K)
        update A, 
        """
        # cal prob of pi:
        DLog.debug('pi Has Nan:', torch.isnan(pi).any())
        logits = torch.log(torch.softmax(pi, dim=-1))
        DLog.debug('logits Has Nan:', torch.isnan(logits).any())

        pi_onehot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)

        # select one component of mu, sig via pi for each node:

        mu_k = torch.sum(mu * pi_onehot, dim=-1) # BN
        sig_k = torch.sum(sig * pi_onehot, dim=-1) #BN

        n = self.get_norm_noise(mu_k.shape[0]) # BN
        DLog.debug('mu shape:', mu_k.shape)
        DLog.debug('sig_k shape:', sig_k.shape)
        DLog.debug('n shape:', n.shape)

        S = mu_k + n*sig_k
        S = S.unsqueeze(dim=-1)
        # change to gumbel softmax, discrete sampling.
        # DLog.debug('S Has Nan:', torch.isnan(S).any())
        Sim = torch.einsum('bnc, bcm -> bnm', S, S.transpose(2, 1)) # need to be softmax

        P = torch.sigmoid(Sim)

        
        pp = torch.stack((P+0.01, 1-P + 0.01), dim=3)
        DLog.debug('min:', torch.min(pp))
        # DLog.debug('max',torch.max(pp))
        pp_logits = torch.log(pp)
        DLog.debug('Has Nan:', torch.isnan(pp_logits).any())
        pp_onehot = F.gumbel_softmax(pp_logits, tau=self.gumbel_tau, hard=False, dim=-1)
        A = pp_onehot[:,:,:,0]
        A = torch.mean(A, dim=0)

        return A

    def forward(self, x, adj_t=None):
        if adj_t is None:
            adj_t = self.adj_fix
            DLog.debug('LGG: adj_t shape', adj_t.shape)
        
        mu = self.mu_nn(x, adj_t)
        sig = self.sig_nn(x, adj_t)
        pi = self.pi_nn(x, adj_t)

        A = self.update_A(mu, sig, pi)

        return A