import os

import torch
import torch.nn.functional as F
from torch import  nn
from torch_scatter import scatter_mean, scatter, scatter_add, scatter_max
from torch_geometric.nn.conv import MessagePassing

import eeg_util
from eeg_util import DLog
from models.baseline_models import GAT
from models.graph_conv_layer import *
from models.encoder_decoder import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class GGN(nn.Module):
    def __init__(self, adj, args, out_mid_features=False):
        super(GGN, self).__init__()
        self.args = args
        self.log = False
        self.adj_eps = 0.1
        self.adj = adj
        self.adj_x = adj

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
            
        elif args.encoder == 'cnn2d':
            cnn = CNN2d(in_dim=args.feature_len, 
                               hid_dim=en_hid_dim, 
                               out_dim=args.decoder_out_dim, 
                               width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.encoder = cnn
            de_out_dim = args.decoder_out_dim
        else:
            self.encoder = MultiEncoders(args, args.feature_len, en_hid_dim, en_out_dim)
            decoder_in_dim = en_out_dim * 2
            de_out_dim = args.decoder_out_dim + decoder_in_dim

        if args.gnn_adj_type == 'rand':
            self.adj = None
            self.adj_tensor = None

        if args.lgg:
            self.LGG = LatentGraphGenerator(args, adj, args.lgg_tau, decoder_in_dim, args.lgg_hid_dim,
                                        args.lgg_k)

        if args.decoder == 'gnn':
            if args.cut_encoder_dim > 0:
                decoder_in_dim *= args.cut_encoder_dim
            self.decoder = GNNDecoder(self.N, args, decoder_in_dim, de_out_dim)
            if args.agg_type == 'cat':
                de_out_dim *= self.N
        elif args.decoder == 'gat_cnn':
            # adj_coo = eeg_util.torch_dense_to_coo_sparse(adj)
            self.adj_x =  torch.ones((self.N, self.N)).float().cuda()
            print('gat adj_x: ', self.adj_x)
            g_pooling = GateGraphPooling(args, self.N)
            gnn = GAT(decoder_in_dim, args.gnn_hid_dim, de_out_dim, 
                               dropout=args.dropout, pooling=g_pooling)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                            width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
            de_out_dim *= 2
            
        elif args.decoder == 'lgg_cnn':
            gnn = GNNDecoder(self.N, args, decoder_in_dim, args.gnn_out_dim)
            cnn_in_dim = decoder_in_dim
            cnn = CNN2d(cnn_in_dim, args.decoder_hid_dim, de_out_dim,
                            width=34, height=self.N, stride=2, layers=3, dropout=args.dropout)
            self.decoder = SpatialDecoder(args, gnn, cnn)
            if args.agg_type == 'cat':
                de_out_dim += args.gnn_out_dim * self.N
            else:
                if args.lgg and args.lgg_time:
                    de_out_dim += args.gnn_out_dim * 34
                else:
                    de_out_dim += args.gnn_out_dim
        else:
            self.decoder = None
            de_out_dim = decoder_in_dim * self.N
            
            
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
                            adj_x = self.LGG(x_t, self.adj)
                    else:
                        adj_x = self.LGG(x_t, self.adj)
                        DLog.debug('Model is Eval!!!!!!!!!!!!!!!!!')
                    adj_x_times.append(adj_x)
                self.adj_x = adj_x_times
            else:
                x_t = x[:, -1, ...]  # NOTE: take last time step.
                if self.training and self.epoch < self.warmup:
                    self.adj_x = self.LGG(x_t, self.adj)
                else:
                    self.adj_x = self.LGG(x_t, self.adj)
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