"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

import torch
import torch.nn as nn
from torch.functional import F
from torch_geometric.nn import GATConv


from eeg_util import DLog

class DiffusionGraphConv(nn.Module):
    def __init__(self, num_supports, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, bias_start=0.0,
                 filter_type='laplacian'):
        """
        Diffusion graph convolution
        Args:
            num_supports: number of supports, 1 for 'laplacian' filter and 2
                for 'dual_random_walk'
            input_dim: input feature dim
            hid_dim: hidden units
            num_nodes: number of nodes in graph
            max_diffusion_step: maximum diffusion step
            output_dim: output feature dim
            filter_type: 'laplacian' for undirected graph, and 'dual_random_walk'
                for directed graph
        """
        super(DiffusionGraphConv, self).__init__()
        num_matrices = num_supports * max_diffusion_step + 1
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._filter_type = filter_type
        self.weight = nn.Parameter(
            torch.FloatTensor(
                size=(
                    self._input_size *
                    num_matrices,
                    output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 1)
        return torch.cat([x, x_], dim=1)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, supports, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes,
        # input_dim/hidden_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        # (batch, num_nodes, input_dim+hidden_dim)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = self._input_size

        x0 = inputs_and_state  # (batch, num_nodes, input_dim+hidden_dim)
        # (batch, 1, num_nodes, input_dim+hidden_dim)
        x = torch.unsqueeze(x0, dim=1)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                # (batch, num_nodes, input_dim+hidden_dim)
                x1 = torch.matmul(support, x0)
                # (batch, ?, num_nodes, input_dim+hidden_dim)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    # (batch, num_nodes, input_dim+hidden_dim)
                    x2 = 2 * torch.matmul(support, x1) - x0
                    x = self._concat(
                        x, x2)  # (batch, ?, num_nodes, input_dim+hidden_dim)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * \
            self._max_diffusion_step + 1  # Adds for x itself
        # (batch, num_nodes, num_matrices, input_hidden_size)
        x = torch.transpose(x, dim0=1, dim1=2)
        # (batch, num_nodes, input_hidden_size, num_matrices)
        x = torch.transpose(x, dim0=2, dim1=3)
        x = torch.reshape(
            x,
            shape=[
                batch_size,
                self._num_nodes,
                input_size *
                num_matrices])
        x = torch.reshape(
            x,
            shape=[
                batch_size *
                self._num_nodes,
                input_size *
                num_matrices])
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(
            self,
            input_dim,
            num_units,
            max_diffusion_step,
            num_nodes,
            filter_type="laplacian",
            nonlinearity='tanh',
            use_gc_for_ru=True):
        """
        Args:
            input_dim: input feature dim
            num_units: number of DCGRU hidden units
            max_diffusion_step: maximum diffusion step
            num_nodes: number of nodes in the graph
            filter_type: 'laplacian' for undirected graph, 'dual_random_walk' for directed graph
            nonlinearity: 'tanh' or 'relu'. Default is 'tanh'
            use_gc_for_ru: decide whether to use graph convolution inside rnn. Default True
        """
        super(DCGRUCell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        if filter_type == "laplacian":  # ChebNet graph conv
            self._num_supports = 1
        elif filter_type == "random_walk":  # Forward random walk
            self._num_supports = 1
        elif filter_type == "dual_random_walk":  # Bidirectional random walk
            self._num_supports = 2
        else:
            self._num_supports = 1

        self.dconv_gate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units * 2,
            filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units,
            filter_type=filter_type)

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        return output_size

    def forward(self, supports, inputs, state):
        """
        Args:
            inputs: (B, num_nodes * input_dim)
            state: (B, num_nodes * num_units)
        Returns:
            output: (B, num_nodes * output_dim)
            state: (B, num_nodes * num_units)
        """
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(supports, inputs, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            value, split_size_or_sections=int(
                output_size / 2), dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))
        # batch_size, self._num_nodes * output_size
        c = self.dconv_candidate(supports, inputs, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c

        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, supports, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self._num_nodes * self._num_units)


def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor
    """
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x)
                     for x in tup)
    else:
        return fn(tup)


def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor
    """
    if isinstance(tups[0], tuple):
        return tuple(
            (torch.cat(
                xs,
                dim) if isinstance(
                xs[0],
                torch.Tensor) else xs[0]) for xs in zip(
                *
                tups))
    else:
        return torch.cat(tups, dim)


class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='laplacian',
                 device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    supports, current_inputs[t, ...], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device)  # (seq_len, batch_size, num_nodes * rnn_units)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device)  # (num_layers, batch_size, num_nodes * rnn_units)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, dcgru_activation=None,
                 filter_type='laplacian', device=None, dropout=0.0):
        super(DCGRUDecoder, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device
        self.dropout = dropout

        cell = DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                         max_diffusion_step=max_diffusion_step,
                         num_nodes=num_nodes, nonlinearity=dcgru_activation,
                         filter_type=filter_type)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            decoding_cells.append(cell)

        self.decoding_cells = nn.ModuleList(decoding_cells)
        self.projection_layer = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)  # dropout before projection layer

    def forward(
            self,
            inputs,
            initial_hidden_state,
            supports,
            teacher_forcing_ratio=None):
        """
        Args:
            inputs: shape (seq_len, batch_size, num_nodes, output_dim)
            initial_hidden_state: the last hidden state of the encoder, shape (num_layers, batch, num_nodes * rnn_units)
            supports: list of supports from laplacian or dual_random_walk filters
            teacher_forcing_ratio: ratio for teacher forcing
        Returns:
            outputs: shape (seq_len, batch_size, num_nodes * output_dim)
        """
        seq_length, batch_size, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        go_symbol = torch.zeros(
            (batch_size,
             self.num_nodes *
             self.output_dim)).to(
            self._device)

        # tensor to store decoder outputs
        outputs = torch.zeros(
            seq_length,
            batch_size,
            self.num_nodes *
            self.output_dim).to(
            self._device)

        current_input = go_symbol  # (batch_size, num_nodes * input_dim)
        for t in range(seq_length):
            next_input_hidden_state = []
            for i_layer in range(0, self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    supports, current_input, hidden_state)
                current_input = output
                next_input_hidden_state.append(hidden_state)
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)

            projected = self.projection_layer(self.dropout(
                output.reshape(batch_size, self.num_nodes, -1)))
            projected = projected.reshape(
                batch_size, self.num_nodes * self.output_dim)
            outputs[t] = projected

            if teacher_forcing_ratio is not None:
                teacher_force = random.random() < teacher_forcing_ratio  # a bool value
                current_input = (inputs[t] if teacher_force else projected)
            else:
                current_input = projected

        return outputs


def last_relevant_pytorch(output, lengths, batch_first=True):
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output


########## Model for seizure classification/detection ##########
class DCRNNModel_classification(nn.Module):
    def __init__(self, args, adj, N, num_classes, in_dim, device=None, out_mid_features=False):
        super(DCRNNModel_classification, self).__init__()

        num_nodes = N
        self.args = args
        self.out_mid_features = out_mid_features
        num_rnn_layers = args.rnn_layer_num
        rnn_units = args.rnn_hidden_len
        enc_input_dim = in_dim
        max_diffusion_step = args.max_diffusion_step

        self.adj = adj
        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.adj_type)

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        input_seq = input_seq.transpose(3, 1)

        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(
            batch_size).to(self._device)

        # last hidden state of the encoder is the context
        # (max_seq_len, batch, rnn_units*num_nodes)
        _, final_hidden = self.encoder(input_seq, init_hidden_state, self.adj)
        # (batch_size, max_seq_len, rnn_units*num_nodes)
        output = torch.transpose(final_hidden, dim0=0, dim1=1)

        # extract last relevant output
        DLog.debug('DCRNNModel_classification output shape:', output.shape)
        # last_out = last_relevant_pytorch(
            # output, torch.ones(batch_size) * 34, batch_first=True)  # (batch_size, rnn_units*num_nodes)
        # (batch_size, num_nodes, rnn_units)
        last_out = output[:, -1, :]
        DLog.debug('DCRNNModel_classification last_out shape:', last_out.shape)

        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        last_out = last_out.to(self._device)
        if self.out_mid_features:
            return last_out
        
        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))
        
        del output

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits
########## Model for seizure classification/detection ##########


########## Model for next time prediction ##########
class DCRNNModel_nextTimePred(nn.Module):
    def __init__(self, args, device=None):
        super(DCRNNModel_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        enc_input_dim = args.input_dim
        dec_input_dim = args.output_dim
        output_dim = args.output_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = args.num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.output_dim = output_dim
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)

        self.encoder = DCRNNEncoder(input_dim=enc_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=rnn_units, num_nodes=num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type)
        self.decoder = DCGRUDecoder(input_dim=dec_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes, hid_dim=rnn_units,
                                    output_dim=output_dim,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=device,
                                    dropout=args.dropout)

    def forward(
            self,
            encoder_inputs,
            decoder_inputs,
            supports,
            batches_seen=None):
        """
        Args:
            encoder_inputs: encoder input sequence, shape (batch, input_seq_len, num_nodes, input_dim)
            encoder_inputs: decoder input sequence, shape (batch, output_seq_len, num_nodes, output_dim)
            supports: list of supports from laplacian or dual_random_walk filters
            batches_seen: number of examples seen so far, for teacher forcing
        Returns:
            outputs: predicted output sequence, shape (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(batch_size).cuda()

        # encoder
        # (num_layers, batch, rnn_units*num_nodes)
        encoder_hidden_state, _ = self.encoder(
            encoder_inputs, init_hidden_state, supports)

        # decoder
        if self.training and self.use_curriculum_learning and (
                batches_seen is not None):
            teacher_forcing_ratio = utils.compute_sampling_threshold(
                self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None
        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio)  # (seq_len, batch_size, num_nodes * output_dim)
        # (seq_len, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs
########## Model for next time prediction ##########


# SeizureNet Implementation without saliency encode.

# <<Automated diagnosis epilepsy seizures with graph generative graph neural networks>>

class DenseLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.5):
        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, hid_dim, kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.Conv2d(hid_dim, hid_dim, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, args, layer_num, in_dim, hid_dim, out_dim, kernel_size=(1, 1)):
        super(DenseBlock, self).__init__()
        self.args = args
        self.downsample = nn.Conv2d(in_dim, hid_dim, kernel_size=kernel_size, stride=1)
        self.layers = nn.ModuleList()
        self.layers.append(DenseLayer(in_dim, hid_dim))
        for i in range(layer_num-2):
            self.layers.append(DenseLayer(hid_dim, hid_dim))
            
        if layer_num >1:
            self.layers.append(DenseLayer(hid_dim, out_dim))
        
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class ConvPool(nn.Module):
    def __init__(self,in_dim, hid_dim):
        super(ConvPool, self).__init__()
        self.conv = nn.Conv2d(in_dim, hid_dim, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(hid_dim)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x
        
        
def conv_L(in_len, kernel, stride, padding=0):
    return int((in_len - kernel + 2 * padding) / stride + 1)


def cal_cnn_outlen(modules, in_len, height=True):
    conv_l = in_len
    pos = 0 if height else 1
    for m in modules:
        if isinstance(m, nn.Conv2d):
            print('m:', m)
            print('kernel_size:',m.kernel_size)
            print('stride:',m.stride)
            print('padding:',m.padding)
            conv_l = conv_L(in_len, m.kernel_size[pos], m.stride[0], m.padding[pos])
            in_len = conv_l
        if not height:
            if isinstance(m, nn.Conv1d):
                conv_l = conv_L(in_len, m.kernel_size[0], m.stride[0], m.padding[0])
                in_len = conv_l
        
        if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d):
            conv_l = conv_L(in_len, m.kernel_size[pos], m.stride, m.padding)
            in_len = conv_l                
    return conv_l


class CNNNet(nn.Module):
    def __init__(self, args, out_mid_features=False):
        super(CNNNet, self).__init__()
        self.args = args
        self.out_mid_features = out_mid_features
        self.dense1 = DenseBlock(args, 3, args.feature_len, 
                                 args.encoder_hid_dim, args.encoder_hid_dim)
        self.conv_pool1 = ConvPool(args.encoder_hid_dim,  args.encoder_hid_dim)
        
        self.dense2 = DenseBlock(args, 3, args.encoder_hid_dim, 
                                 args.encoder_hid_dim, args.encoder_hid_dim, kernel_size=(1, 3))
        
        self.conv_pool2 = ConvPool(args.encoder_hid_dim,  args.encoder_hid_dim)
        
        # self.dense3 = DenseBlock(args, 3, args.encoder_hid_dim, 
        #                          args.encoder_hid_dim, args.encoder_hid_dim)
        # self.conv_pool3 = ConvPool(args.encoder_hid_dim,  args.encoder_hid_dim)
        
        H = cal_cnn_outlen(self.dense1.modules(), 34)
        print('H1:', H) # 28
        W = cal_cnn_outlen(self.dense1.modules(), 20, height=False)
        print('W:', W) # 14
        
        out_dim = 2048
        
        self.fc = nn.Linear(out_dim, args.predict_class_num)

    def forward(self, x):
        DLog.debug('SeizureNet input shape:', x.shape)
        x = self.dense1(x)
        x = self.conv_pool1(x)
        DLog.debug('SeizureNet conv_pool1 output shape:', x.shape)
        
        x = self.dense2(x)
        # x = self.conv_pool2(x)
        DLog.debug('SeizureNet conv_pool2 output shape:', x.shape)
        x = torch.flatten(x, start_dim=1)
        if self.out_mid_features:
            return x
        x = self.fc(x)
        DLog.debug('SeizureNet output shape:', x.shape)
        
        # x = self.dense3(x)
        # x = self.conv_pool3(x)
        # DLog.debug('SeizureNet conv_pool3 output shape:', x.shape)
        
        return x
        

##-------------------------- Transformer :  https://arxiv.org/pdf/1910.05448.pdf  --------------- ##


from torch.nn import Module
import torch
import math
import torch.nn.functional as F


class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, stage):

        residual = x
        x, score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score


class FeedForward(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, x):

        x = self.linear_1(x)
        x = F.relu(x)
        x = self.linear_2(x)

        return x
    
class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage):
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2 ** 32 + 1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score
    

class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_hz: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False,
                out_mid_features: bool = False):
        super(Transformer, self).__init__()
        self.out_mid_features = out_mid_features
        self.encoder_list_1 = nn.ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = nn.ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])
        self.encoder_list_3 = nn.ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])
        #         step,cha,hz

        self.embedding_input = torch.nn.Linear(d_channel * d_hz, d_model)
        self.embedding_channel = torch.nn.Linear(d_input * d_hz, d_model)
        self.embedding_hz = torch.nn.Linear(d_input * d_channel, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel + d_model * d_hz, 3)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel + d_model * d_hz, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage='test'):
        """
        前向传播
        :param x: 输入
        :param stage: 用于描述此时是训练集的训练过程还是测试集的测试过程  测试过程中均不在加mask机制
        :return: 输出，gate之后的二维向量，step-wise encoder中的score矩阵，channel-wise encoder中的score矩阵，step-wise embedding后的三维矩阵，channel-wise embedding后的三维矩阵，gate
        """
        # step-wise
        # score矩阵为 input， 默认加mask 和 pe
        # 
        DLog.debug('input x shape:', x.shape) # BCNT
        x = x.transpose(3, 1)


        step_x = x
        step_x = step_x.reshape(step_x.shape[0], step_x.shape[1], step_x.shape[2] * step_x.shape[3]) # BT NC??
        encoding_1 = self.embedding_input(step_x)
        input_to_gather = encoding_1

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe
        # 样本 时间 通道 赫兹
        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # score矩阵为channel 默认不加mask和pe
        channel_x = x
        channel_x = channel_x.transpose(2, 1)
        channel_x = channel_x.reshape(channel_x.shape[0], channel_x.shape[1], channel_x.shape[2] * channel_x.shape[3])
        encoding_2 = self.embedding_channel(channel_x)
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # hz-wise
        hz_x = x
        hz_x = hz_x.transpose(3, 1)
        hz_x = hz_x.reshape(hz_x.shape[0], hz_x.shape[1], hz_x.shape[2] * hz_x.shape[3])
        encoding_3 = self.embedding_hz(hz_x)
        channel_to_gather = encoding_3

        for encoder in self.encoder_list_3:
            encoding_3, score_channel = encoder(encoding_3, stage)

        # 三维变二维
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)
        encoding_3 = encoding_3.reshape(encoding_3.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2, encoding_3], dim=-1)), dim=-1)
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2], encoding_3 * gate[:, 2:3]], dim=-1)

        if self.out_mid_features:
            return encoding

        # 输出
        output = self.output_linear(encoding)

        del encoding
        del encoding_1
        del encoding_2
        del encoding_3
        del input_to_gather
        del channel_to_gather
        del gate
        

        return output

        # return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
        
        
"""
Graph attention network implementation similar to https://arxiv.org/abs/1710.10903.
"""
class GraphAttentionLayer(nn.Module):
    """
    Original implementation cannot support minibatch x, we re-implement it to support.
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = None

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # re-implement it to support batch:
        Wh = torch.matmul(h, self.W) # h.shape: (B, N, in_features), Wh.shape: (B, N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        self.attention = attention

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

"""
Multi-head GAT
"""
class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.6, alpha=0.2, nheads=8, pooling=None):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.g_pooling = pooling
        self.attentions = [GraphAttentionLayer(in_dim, hid_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        self.out_att = GraphAttentionLayer(hid_dim * nheads, out_dim, dropout=dropout, alpha=alpha, concat=True)

    def forward(self, adj, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # concate node features from different heads.
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj)) # BNC
        if self.g_pooling is not None:
            x = self.g_pooling(x) # BC
        return x