import torch
import torch.nn.functional as F
from torch import  nn

from eeg_util import DLog

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
        for gnn in self.gnns:
            x = gnn(x, adj)
        DLog.debug('MultilayerGNN out shape:', x.shape)
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
        if self.args.agg_type == 'gate':
            x = torch.einsum('bnc, n -> bc', x, self.gate)
        elif self.args.agg_type == 'cat':
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
        