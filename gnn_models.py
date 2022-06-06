# using GCN to predict:
from torch_geometric.utils import add_self_loops, degree

import numpy as np
import networkx as nx

import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

import math

import matplotlib.pyplot as plt



def get_spd_matrix(G, S, max_spd=5):
    spd_matrix = np.zeros((G.number_of_nodes(), len(S)), dtype=np.float32)
    for i, node_S in enumerate(S):
        for node, length in nx.shortest_path_length(G, source=node_S).items():
            spd_matrix[node, i] = min(length, max_spd)
    return spd_matrix


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Valid: {result[:, 0].max():.2f}')
            print(f'   Final Test: {result[argmax, 1]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))
            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Valid: {r.mean():.4f} +- {r.std():.4f}')
            r = best_result[:, 1]
            print(f'   Final Test: {r.mean():.4f} +- {r.std():.4f}')


class SAGEConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        # if isinstance(edge_index, Tensor):
        #     assert edge_attr is not None
        #     assert x[0].size(-1) == edge_attr.size(-1)
        # elif isinstance(edge_index, SparseTensor):
        #     assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_attr is None:
            return F.relu(x_j)
        return F.relu(x_j + edge_attr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GraphSAGE,self).__init__()
        self.convs = torch.nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr=None, emb_ea=None):
        if edge_attr is not None:
            edge_attr = torch.mm(edge_attr, emb_ea)

        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, edge_attr)  # no nonlinearity
        return x


def gumbel_sampling(shape, mu=0, beta=1):
    y = torch.rand(shape).cuda() + 1e-20  # ensure all y is positive.
    g = mu - beta * torch.log(-torch.log(y)+1e-20)
    return g


class LinkPredictorMy(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, num_gumbels=1, emperical=False):
        super(LinkPredictorMy, self).__init__()

        self.W = nn.Parameter(torch.randn(dim_in, dim_in))
        self.tau = 0.5
        self.dim_hidden = dim_hidden
        self.linkpred = LinkPredictor(dim_in, dim_in, 1, 2, 0.3)

        self.warm_up = -1
        self.count = 0

        self.num_gumbels = num_gumbels
        self.emperical = emperical


    def expectation_sampling(self, pi):
        U = torch.log(pi)
        gumbs = gumbel_sampling(pi.shape)
        for n in range(self.num_gumbels-1):
            gumbs += gumbel_sampling(pi.shape)
        U += gumbs/self.num_gumbels
        U = U/self.tau
        return U

    def forward(self, zi, zj):
        """
        zi, zj \in (batch_size, dim_in)
        """
        def hasNan(x):
            return torch.isnan(x).any()

        if self.count < self.warm_up:
            self.count += 1
            return self.linkpred(zi, zj)

        P = torch.sigmoid(torch.einsum('nc, nc -> n', zi@self.W, zj))
        # P = F.dropout(P, p=0.2)
        pi = torch.stack((1-P, P),dim=1)
        if self.emperical:
            U = self.expectation_sampling(pi)
        else:
            # TODO: add one gumbel:
            U = torch.log(pi)
            U = U/self.tau
        # TODO: replace the softmax to sigmoid???
        p_exp = torch.exp(U)
        p_sum = torch.sum(p_exp, dim=1)

        p1 = p_exp[:,1]/p_sum
        # edge = F.softmax(U, dim=1)
        if hasNan(p1):
            print('hasNan:p1p1p1p1p:', p1)
        return p1

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.linkpred.reset_parameters()
        

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.3):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class GCN(MessagePassing):
    def __init__(self, in_channel, out_channel, aggr="add", flow: str = "source_to_target", node_dim: int = -2):
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim)
        self.lin = nn.Linear(in_channel, out_channel)
    
    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)
        if isinstance(x, Tensor):
            xx: OptPairTensor = (x, x)
        
        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=xx, edge_attr=edge_attr, norm=norm)
        out = out + xx[1]
        # Step 2: Linearly transform node feature matrix.
        # out = F.relu(out)
        
        return out

    def message(self, x_j, edge_attr, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * (x_j + edge_attr)
        
class MultiGCN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers):
        super(MultiGCN,self).__init__()
        self.convs = nn.ModuleList()

        self.convs.append(GCN(dim_in, dim_hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCN(dim_hidden, dim_hidden))
        self.convs.append(GCN(dim_hidden, dim_out))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr, emb_ea):
        edge_attr = torch.mm(edge_attr, emb_ea)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, adj_t, edge_attr)  # no nonlinearity
        return x     


    
class GenerativeGNN(nn.Module):
    def __init__(self, A_0, tau, dim_in, dim_hidden, dim_out, num_layers):
        super(GenerativeGNN,self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCN(dim_in, dim_hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCN(dim_hidden, dim_hidden))
        self.convs.append(GCN(dim_hidden, dim_out))

        self.W = nn.Parameter(torch.rand(dim_in, dim_in))
        self.tau = tau
        self.A_0 = A_0
        self.N = A_0.size(0) # num of nodes.
        self.dim_hidden = dim_hidden


    def update_A(self, z, edge_index, edge_attr):
        """
        update A via node representation Z (N x N)
        """
        P = F.sigmoid(torch.einsum('nc, nc -> n', z@self.W, z))
        pi = torch.tensor(torch.stack((1-P, P),dim=2))
        print('pi shape: ', pi.shape)
        U = torch.log(pi) + gumbel_sampling(pi.shape)
        U = U/self.tau
        # TODO: Gumbel softmax sampling:
        A = F.softmax(U, dim=2)
        self.A_opt = A


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr, emb_ea):
        edge_attr = torch.mm(edge_attr, emb_ea)
        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, adj_t, edge_attr)  # no nonlinearity
        return x


class VGAE(nn.Module):
	def __init__(self, adj, args):
		super(VGAE, self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0), self.args.hidden2_dim)
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		return sampled_z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred


class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation = F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim) 
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x,self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)


class GAE(nn.Module):
	def __init__(self,adj,args):
		super(GAE,self).__init__()
		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		z = self.mean = self.gcn_mean(hidden)
		return z

	def forward(self, X):
		Z = self.encode(X)
		A_pred = dot_product_decode(Z)
		return A_pred

def update_A(z, W=None):
    """
    update A via node representation Z (N x N)
    """
    P = F.sigmoid(torch.einsum('nc, nc -> n', z@W, z))
    print('P shape:', P.shape)
    pi = torch.tensor(torch.stack((1-P, P),dim=1))
    print('pi shape: ', pi.shape)
    U = torch.log(pi) + gumbel_sampling(pi.shape)
    # plot:


    # TODO: Gumbel softmax sampling:
    A = F.softmax(U, dim=1)
    print('A : ', A[0, :])



if __name__ == '__main__':
    
    z = torch.rand((1000, 2))
    
    plt.hist(z[:,0])
    
    w = torch.nn.Parameter(torch.ones((2, 2)))
    # nn.init.kaiming_uniform_(w, a=math.sqrt(5))


    print('z shape:', z.shape)
    print('w shape:', w.shape)
    update_A(z, w)










