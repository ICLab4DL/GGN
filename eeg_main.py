# -*- coding: utf-8 -*

import random
import collections
import time
from os import walk
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch import nn

from models.ggn import GGN
from eeg_util import *
import eeg_util
from models.baseline_models import *


import networkx as nx
import json


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])


# NOTE: FIXED, Order cannot be changed!!!!!!!!!
s_types = {'FNSZ': 1836,'GNSZ': 583, 'CPSZ': 367,'ABSZ': 99,  'TNSZ': 62, 'SPSZ': 52, 'TCSZ': 48}
label_dict = {}
number_label_dict = {}
for i, k in enumerate(s_types.keys()):
    label_dict[k] = i
    number_label_dict[i] = k
print('labels:', label_dict)

def load_data(args, dataset_dir, batch_size, shuffle=False):
    if not os.path.exists(dataset_dir):
        print(dataset_dir, " is not exist!!")
        return
    
    feature = np.load(os.path.join(dataset_dir, "feature.npy"))
    label = np.load(os.path.join(dataset_dir, "label.npy"))
    print(f'{args.dataset} label shape: ', label.shape)
    print(f'{args.dataset} feature shape', feature.shape)
    
    return feature, label


def load_tuh_data(args, feature_name=""):

    feature = np.load(os.path.join(args.data_path, f"seizure_x{feature_name}.npy"))
    label = np.load(os.path.join(args.data_path, f"seizure_y{feature_name}.npy"))
    print('load seizure data, shape:', feature.shape, label.shape)


    if args.testing:
        shuffled_index = np.load('shuffled_index.npy')
    else:
        # shuffle:
        shuffled_index = np.random.permutation(np.arange(feature.shape[0]))
        np.save('./shuffled_index', shuffled_index)
        print('saved shuffle index')
    print('shuffled_index:', shuffled_index)
    
    
    feature = feature[shuffled_index]
    label = label[shuffled_index]


    # train, test:

    label_dict = {}
    for i, l in enumerate(label):
        if l not in label_dict:
            label_dict[l] = []
        label_dict[l].append(i)


    # Filter the MYSZ:

    # take 1/3 as test set for each seizure type.
    train_x, train_y, test_x, test_y = [],[],[],[]
    for k, v in label_dict.items():
        test_size = int(len(v)/3)
        train_x.append(feature[v[test_size:]])
        train_y.append(label[v[test_size:]])
        test_x.append(feature[v[:test_size]])
        test_y.append(label[v[:test_size]])
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    print('before trans:', train_x.shape,train_y.shape,test_x.shape,test_y.shape)
    # reshape to B, C, N, T:
    B, T, N, C = train_x.shape
    train_x =  train_x.transpose(0, 3, 2, 1)
    test_x = test_x.transpose(0, 3, 2, 1)

    print('after trans:', train_x.shape,train_y.shape,test_x.shape,test_y.shape)

    # load to dataloader:
    return [train_x, test_x], [train_y, test_y]


def generate_tuh_data(args, file_name=""):
    data_path = args.data_path

    freqs = [12]
    x_data = []
    y_data = []

    types_dict = {}
    for freq in freqs:
        x_f_data = []
        y_f_data = []
        min_len = 10000
        freq_file_name = f"fft_seizures_wl1_ws_0.25_sf_250_fft_min_1_fft_max_{freq}"
        dir, _, files = next(walk(os.path.join(data_path, freq_file_name)))
        for i, name in enumerate(files):
            fft_data = pickle.load(open(os.path.join(dir,name), 'rb'))
            if fft_data.seizure_type == 'MYSZ':
                continue
            if fft_data.data.shape[0] < 34:
                continue
            if fft_data.data.shape[0] < min_len:
                min_len = fft_data.data.shape[0]
                
            x_f_data.append(fft_data.data)
            y_f_data.append(label_dict[fft_data.seizure_type])
        print('min len:', min_len)
        x_f_data = [d[:min_len,...] for d in x_f_data]
        x_f_data = np.stack(x_f_data, axis=0)
        print(x_f_data.shape)
        y_f_data = np.stack(y_f_data, axis=0)
        print(y_f_data.shape)
        x_data.append(x_f_data)
        y_data.append(y_f_data)

    # check each y_f_data:
    print('prepare save!')
    x_data = np.concatenate(x_data, axis=3)
    print('x data shape:', x_data.shape)
    np.save(f'seizure_x_{file_name}.npy', x_data)
    np.save(f'seizure_y_{file_name}.npy', y_data[0])
    print('y data shape:', y_data[0].shape)
    print('save done!')


def transform_SEED(dataset_dir, batch_size):
    import glob

    features = []
    labels = []
    if not os.path.exists(dataset_dir):
        print(dataset_dir, " is not exist!!")
        return

    dataset_dir += "/*"
    print(dataset_dir)
    for dir in glob.iglob(dataset_dir):
        datasets = {}
        feature = np.load(os.path.join(dir, "feature.npy"))
        label = np.load(os.path.join(dir, "label.npy"))
        features.append(feature)
        print(feature.shape)
        # label: -1, 0, 1 --->  0, 1, 2
        labels.append(label + 1)
    feature = np.stack(features, axis=0)
    label = np.stack(labels, axis=0)[:,:,0,0]
    print('SEED feature:', feature.shape)
    print('SEED labels:', label.shape)
    
    np.save('/li_zhengdao/github/EEG/data/SEED/de_LDS/feature.npy', feature)
    np.save('/li_zhengdao/github/EEG/data/SEED/de_LDS/label.npy', label)


def normalize_seizure_features(features):
    """inplace-norm
    Args:
        features (list of tensors): train,test,val
    """
    for i in range(len(features)):
        # (B, F, N, T)
        for j in range(features[i].shape[-1]):
            features[i][..., j] = normalize(features[i][..., j])
    
def generate_dataloader_seizure(features, labels, args):
    """
     features: [train, test, val], if val is empty then val == test
     train: B, T, N, F(12,24,48,64,96)
    """
    cates = ['train', 'test', 'val']
    datasets = dict()
    # normalize over feature dimension

    for i in range(len(features)):
        datasets[cates[i] + '_loader'] = SeqDataLoader(features[i], labels[i], args.batch_size, cuda=args.cuda)

    if len(features) < 3: # take test as validation.
        datasets['val_loader'] = SeqDataLoader(features[-1], labels[-1], args.batch_size, cuda=args.cuda)

    return datasets


def generate_dataloader(features, labels, args, independent, subj_id=14, shuffle=False):
    # features: (Subject, Trials, N, Seq, Channel)
    datasets = dict()
    batch_size = args.batch_size
    # if subject independent, then leave one subject as the test set, do the cross-validaton.
    # and one subject as the val.


    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    test_features = []
    test_labels = []
    seq_len = args.eeg_seq_len

    # filtering:
    # cut the seq:
    features = features[:, :, :, :seq_len, :]


    if independent:
        print('subject independent')
        val_index = subj_id
        print('the val subject index: ', val_index)
        print('the test subject index: ', subj_id)

        for i in range(len(features)):
            if i == subj_id:
                test_features.append(features[i])
                test_labels.append(labels[i])
                val_features.append(features[i])
                val_labels.append(labels[i])
                print('test and val')
            elif i == val_index:
                val_features.append(features[i])
                val_labels.append(labels[i])
                print('only val')
            else:
                train_features.append(features[i])
                train_labels.append(labels[i])
    else:
        for i in range(len(features)):
            train_features.append(features[i][:27, ...])
            train_labels.append(labels[i][:27, ...])
            
            val_features.append(features[i][27:, ...])
            val_labels.append(labels[i][27:, ...])
            
            test_features.append(features[i][27:, ...])
            test_labels.append(labels[i][27:,...])

    train_features = np.stack(train_features, axis=0).reshape(-1, 62, seq_len, 5)
    print('train_features :', train_features.shape)
    train_labels = np.stack(train_labels, axis=0).reshape(-1, 1)
    print('train_labels :', train_labels.shape)

    val_features = np.stack(val_features, axis=0).reshape(-1, 62, seq_len, 5)
    print('val_features :', val_features.shape)
    val_labels = np.stack(val_labels, axis=0).reshape(-1, 1)
    print('val_labels :', val_labels.shape)

    test_features = np.stack(test_features, axis=0).reshape(-1, 62, seq_len, 5)
    print('test_features :', test_features.shape)

    test_labels = np.stack(test_labels, axis=0).reshape(-1, 1)
    print('test_labels :', test_labels.shape)

    #     features = np.stack(features, axis=0).reshape(15 * 45, 62, 300, 5)

    #     labels = np.stack(labels, axis=0)[..., 0, 0].reshape(15 * 45, 1)  # each 62 channels have same labels.

    # normalize all features
    for i in range(5):
        train_features[..., i] = normalize(train_features[..., i])
        val_features[..., i] = normalize(val_features[..., i])
        test_features[..., i] = normalize(test_features[..., i])

    # temporally cut down seq to 210.
    train_features = train_features[:, :, :args.eeg_seq_len, :]
    val_features = val_features[:, :, :args.eeg_seq_len, :]
    test_features = test_features[:, :, :args.eeg_seq_len, :]

    print("cut features shape:", train_features.shape)
    # shape (samples, features, node, sequence)
    train_features = np.transpose(train_features, (0, 3, 1, 2))
    val_features = np.transpose(val_features, (0, 3, 1, 2))
    test_features = np.transpose(test_features, (0, 3, 1, 2))

    def shuffle_f(features, labels):
        num_samples = features.shape[0]
        p = np.random.permutation(num_samples)
        return features[p], labels[p]

    # shuffle all samples
    if shuffle:
        train_features, train_labels = shuffle_f(train_features, train_labels)
        val_features, val_labels = shuffle_f(val_features, val_labels)
        test_features, test_labels = shuffle_f(test_features, test_labels)

    print("train features shape: ", train_features.shape)
    print("train labels shape: ", train_labels.shape)

    # add additional training dataloader as validation:
    for category in ['train', 'val', 'test']:
        datasets[category + '_loader'] = SeqDataLoader(locals()[category + "_features"], locals()[category + "_labels"],
                                                       args.batch_size)
        if category == 'train':
            datasets['train_as_val'] = SeqDataLoader(locals()[category + "_features"], locals()[category + "_labels"],
                                                     args.batch_size)

    return datasets


def init_adjs(args, index=0):
    using_corr=False
    adjs = []
    if using_corr:
        chans = datasets['train_loader'].xs
        print(chans.shape) #  405, 5, 62, 210
        adjs = []
        for c in range(1):
            normed_chans = []
            for s in range(0, 15, 15):
                for i in range(0, 9, 9):
                    normed_chans.append(chans[s+i, c,:,:].transpose(1, 0).cpu().detach().numpy())
            normed_chans_len = len(normed_chans)
            mean_corr = sum(get_corrs([global_mmnorm(nc) for nc in normed_chans]))/ normed_chans_len
            adjs.append(mean_corr)
        print('adjs len: ', len(adjs))
        print(adjs[0])
    else:
        if args.adj_type == 'rand10':
            adj_mx = eeg_util.generate_rand_adj(0.1*(index+1), N=20)
        elif args.adj_type == 'er':
            adj_mx = nx.to_numpy_array(nx.erdos_renyi_graph(20, 0.1*(index+1)))
        else:
            adj_mx = load_eeg_adj(args.adj_file, args.adj_type)
        adjs.append(adj_mx)

    #     model = EEGEncoder(adj_mx, args, is_gpu=args.cuda)
    adj = torch.from_numpy(adjs[0]).float().cuda()
    adjs[0] = adj
    return adjs


def chose_model(args, adjs):
    if args.task.upper() == 'GGN':
        adj = adjs[0]
        model = GGN(adj, args)
    elif args.task == 'transformer':
        DEVICE = torch.device("cuda:0" if args.cuda else "cpu")  
        print(f'use device: {DEVICE}')
        models = 512
        hiddens = 1024
        q = 8
        v = 8
        h = 8
        N = 8
        dropout = 0.2
        pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
        mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask

        inputs = 34
        channels = 20
        outputs = args.predict_class_num  # 分类类别
        hz = args.feature_len
        model = Transformer(d_model=models, d_input=inputs, d_channel=channels, d_hz = hz, d_output=outputs, d_hidden=hiddens,
                        q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE)
    elif args.task == 'gnnnet':
        model = DCRNNModel_classification(
        args, adjs, adjs[0].shape[0], args.predict_class_num, args.feature_len, device='cuda')
    elif args.task == 'cnnnet':
        model = CNNNet(args)
    else:
        model = None
        print('No model found!!!!')
    return model



def init_trainer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    def lr_adjust(epoch):
        if epoch < 20:
            return 1
        
        return args.lr_decay_rate ** ((epoch - 19) / 3 + 1)
    
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_adjust)
    
    c={0: 1224, 1: 389, 2: 245, 3: 66, 4: 42, 5: 35, 6: 32}
    w = np.array([c[i] for i in range(7)])
    m = np.median(w)
    total = np.sum(w)
    weights = None
    if args.weighted_ce == 'prop':
        weights =1 - w/total
    elif args.weighted_ce == 'rand':
        weights = np.random.rand(7)*10
    elif args.weighted_ce == 'median':
        weights = m/w
    if weights is not None:
        weights =  torch.from_numpy(weights).float().cuda()
    print('weights:', weights)

    if args.focalloss:
        crite = FocalLoss(nn.CrossEntropyLoss(weight=weights, reduce=False), alpha=0.9, gamma=args.focal_gamma)
    else:
        crite = nn.CrossEntropyLoss(weight=weights)
        
    trainer = Trainer(args, model, optimizer, criterion=crite, sched=lr_sched)
    return trainer

def train_eeg(args, datasets, index=0):
    # SummaryWriter

    import os
    dt = time.strftime("%m_%d_%H_%M", time.localtime())
    log_dir = "./tfboard/"+args.server_tag+"/" + dt
    print('tensorboard path:', log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)
    
    adjs = init_adjs(args, index)

    model = chose_model(args, adjs)

    print('args_cuda:', args.cuda)
    if args.cuda:
        print('rnn_train RNNBlock to cuda!')
        model.cuda()
    else:
        print('rnn_train RNNBlock to cpu!')

    # add scheduler.
    trainer = init_trainer(model, args)
    
    best_val_acc = 0
    best_unchanged_threshold = 100  # accumulated epochs of best val_mae unchanged
    best_count = 0
    best_index = -1
    train_val_metrics = []
    start_time = time.time()
    basedir, file_tag = os.path.split(args.best_model_save_path)
    model_save_path = os.path.join(basedir, f'{index}_{file_tag}')
    
    for e in range(args.epochs):
        datasets['train_loader'].shuffle()
        train_loss, train_preds = [], []

        for i, (input_data, target) in enumerate(datasets['train_loader'].get_iterator()):
            loss, preds = trainer.train(input_data, target)
            # training metrics
            train_loss.append(loss)
            train_preds.append(preds)
        # validation metrics
        val_loss, val_preds = [], []
    
        for j, (input_data, target) in enumerate(datasets['val_loader'].get_iterator()):
            loss, preds  = trainer.eval(input_data, target)
            # add metrics
            val_loss.append(loss)
            val_preds.append(preds)

        # cal metrics as a whole:
        # reshape:
        train_preds = torch.cat(train_preds, dim=0)
        val_preds = torch.cat(val_preds, dim=0)
        
        train_acc = eeg_util.calc_eeg_accuracy(train_preds, datasets['train_loader'].ys)
        val_acc = eeg_util.calc_eeg_accuracy(val_preds, datasets['val_loader'].ys)

        m = dict(train_loss=np.mean(train_loss), train_acc=train_acc,
                 val_loss=np.mean(val_loss), val_acc=val_acc)

        m = pd.Series(m)

        if e % 20 == 0:
            print('epoch:', e)
            print(m)
        # write to tensorboard:
        writer.add_scalars(f'epoch/loss', {'train': m['train_loss'], 'val': m['val_loss']}, e)
        writer.add_scalars(f'epoch/acc', {'train': m['train_acc'], 'val': m['val_acc']}, e)

        train_val_metrics.append(m)
        if m['val_acc'] > best_val_acc:
            best_val_acc = m['val_acc']
            best_count = 0
            print("update best model, epoch: ", e)
            torch.save(trainer.model.state_dict(), model_save_path)
            print(m)
            best_index = e
        else:
            best_count += 1
        if best_count > best_unchanged_threshold:
            print('Got best')
            break

        trainer.lr_schedule()
    print('training: :')
    if args.lgg:
        print('after training adj_fix', trainer.model.LGG.adj_fix[0])
    print('best_epoch:', best_index)

    test_model = chose_model(args, adjs)
    test_model.load_state_dict(torch.load(model_save_path))
    test_model.cuda()
    trainer.model = test_model
    if args.lgg:
        print('after load best model adj_fix', trainer.model.LGG.adj_fix[0])
    
    test_metrics = []
    test_loss, test_preds = [], []


    for i, (input_data, target) in enumerate(datasets['test_loader'].get_iterator()):
        loss, preds = trainer.eval(input_data, target)
        # add metrics
        test_loss.append(loss)
        test_preds.append(preds)
    # cal metrics as a whole:

    # reshape:
    test_preds = torch.cat(test_preds, dim=0)
    test_preds = torch.softmax(test_preds, dim=1)

    test_acc = eeg_util.calc_eeg_accuracy(test_preds, datasets['test_loader'].ys)

    m = dict(test_acc=test_acc, test_loss=np.mean(test_loss))
    m = pd.Series(m)
    print("test:")
    print(m)
    test_metrics.append(m)
    preds_b = test_preds.argmax(dim=1)
    
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    confused_fig_dir = os.path.join(fig_save_dir, f'{file_tag}_{index}_confusion.png')
    loss_fig_dir = os.path.join(basedir, date_dir, f'{file_tag}_{index}_loss.png')
    
    plot_confused_cal_f1(preds_b, datasets['test_loader'].ys, fig_dir=confused_fig_dir)
    plot(train_val_metrics, test_metrics, loss_fig_dir)

    print('finish rnn_train!, time cost:', time.time() - start_time)
    return train_val_metrics, test_metrics


def cal_f1(preds, labels):

    mi_f1 = f1_score(labels, preds, average='micro')
    ma_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')


    return mi_f1, ma_f1, weighted_f1

def plot_confused_cal_f1(preds, labels, fig_dir):
    preds = preds.cpu()
    labels = labels.cpu()
    
    ori_preds = preds
    sns.set()
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.gca()
    gts = [number_label_dict[int(l)][:-2] for l in labels]
    preds = [number_label_dict[int(l)][:-2] for l in preds]
    
    label_names = [v[:-2] for v in number_label_dict.values()]
    print(label_names)
    C2= np.around(confusion_matrix(gts, preds, labels=label_names, normalize='true'), decimals=2)

    # from confusion to ACC, micro-F1, macro-F1, weighted-f1.
    print('Confusion:', C2)
    mi_f1, ma_f1, w_f1 = cal_f1(ori_preds, labels)
    print(f'micro f1: {mi_f1}, macro f1: {ma_f1}, weighted f1: {w_f1}')

    sns.heatmap(C2, cbar=True, annot=True, ax=ax, cmap="YlGnBu", square=True,annot_kws={"size":9},
        yticklabels=label_names,xticklabels=label_names)

    ax.figure.savefig(fig_dir, transparent=False, bbox_inches='tight')


def plot(train_val_metrics, test_metrics, fig_filename='mae'):
    epochs = len(train_val_metrics)
    x = range(epochs)
    train_loss = [m['train_loss'] for m in train_val_metrics]
    val_loss = [m['val_loss'] for m in train_val_metrics]

    plt.figure(figsize=(8, 6))
    plt.plot(x, train_loss, '', label='train_loss')
    plt.plot(x, val_loss, '', label='val_loss')
    plt.title('loss')
    plt.legend(loc='upper right')  # 设置label标记的显示位置
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_filename)
    

def multi_train(args, tags="", runs=10):
    '''
    train multiple times, analyze the results, get the mean and variance.
    '''

    test_loss = []
    test_acc = []
    xs, ys = load_tuh_data(args)
    normalize_seizure_features(xs)
    datasets = generate_dataloader_seizure(xs,ys,args)
    
    for i in range(runs):
        if args.dataset != 'TUH':
            features, labels = load_data(args, args.data_path, args.batch_size)
            datasets = generate_dataloader(features, labels, args, args.independent, subj_id=i)
        
        tr, te = train_eeg(args, datasets, i)
        test_loss.append(te[0]['test_loss'])
        test_acc.append(te[0]['test_acc'])

    # Analysis:
    test_loss_m = np.mean(test_loss)
    test_loss_v = np.std(test_loss)

    test_acc_m = np.mean(test_acc)
    test_acc_v = np.std(test_acc)

    print('%s,trials: %s, t loss mean/std: %f/%f, t acc mean/std: %f%s/%f \n' % (
        tags, runs, test_loss_m, test_loss_v, test_acc_m, '%', test_acc_v))


def testing(args, dataloaders, test_model, batch=False):
    torch.cuda.empty_cache()
    test_model.cuda()
    preds = []
    for x, y in dataloaders['test_loader'].get_iterator():
        p = test_model(x)
        preds.append(p.detach().cpu())
        del p
        torch.cuda.empty_cache()

    preds = torch.cat(preds, dim=0)
        
    print('preds shape:', preds.shape)
    preds = torch.softmax(preds, dim=1)
    
    basedir, file_tag = os.path.split(args.fig_filename)
    date_dir = time.strftime('%Y%m%d', time.localtime(time.time()))
    fig_save_dir = os.path.join(basedir, date_dir)
    if not os.path.exists(fig_save_dir):
        os.makedirs(fig_save_dir)
    confused_fig_dir = os.path.join(fig_save_dir, f'testing_confusion_map_{file_tag}.png')
    
    preds_b = preds.argmax(dim=1)
    plot_confused_cal_f1(preds_b, datasets['test_loader'].ys, fig_dir=confused_fig_dir)
    
    return preds


if __name__ == "__main__":
    
    start_t = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    args = eeg_util.get_common_args()
    args = args.parse_args()
    eeg_util.DLog.init(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)



    if args.testing:
        print('Unit_test!!!!!!!!!!!!!')
        if args.arg_file != 'None':
            args_dict = vars(args)
            print(args_dict.keys())
            print('testing args:')
            with open(args.arg_file, 'rt') as f:
                args_dict.update(json.load(f))
            print('args_dict keys after update:', args_dict.keys())
            args.testing = True
            
        xs, ys = load_tuh_data(args)
        normalize_seizure_features(xs)
        datasets = generate_dataloader_seizure(xs,ys,args)
        adjs = init_adjs(args)
        test_model = chose_model(args, adjs)
        test_model.load_state_dict(torch.load(args.best_model_save_path), strict=False)
        test_model.cuda()
        test_model.eval()
        DLog.log('args is : by DLOG:', args)
        testing(args, datasets, test_model)
        
    elif args.task == 'generate_data':
        generate_tuh_data(args, file_name="from_begin")
    else:
        dt = time.strftime('%Y%m%d', time.localtime(time.time()))
        model_used = "basic model"
        
        tags = "type:" + model_used + str(dt)
        # Save the args:
        _, file_tag = os.path.split(args.fig_filename)
        args_path = f'./args/{dt}/'
        if not os.path.exists(args_path):
            os.makedirs(args_path)
        with open(os.path.join(args_path, f'{file_tag}.json'), 'wt') as f:
            json.dump(vars(args), f, indent=4)
            
        DLog.log('------------ Args Saved! -------------')
        DLog.log('args is : by DLOG:', args)
        multi_train(args, tags=tags, runs=args.runs)
        
    print('Main running Over, total time spent:',time.time() - start_t)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
