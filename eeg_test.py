# -*- coding: utf-8 -*

import random
import time
import copy

import matplotlib.pyplot as plt
from scipy.sparse import data
from torch.utils.tensorboard import SummaryWriter

from seizure_models import SEEGModel, Trainer
from eeg_util import *
from torch.autograd import Variable
import seaborn as sns
from sklearn.metrics import confusion_matrix
import collections
from os import walk
import os
import eeg_util
from torch import optim
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

args = Args()

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


def load_tuh_data(args):

    feature = np.load(os.path.join(args.data_path, "seizure_x.npy"))
    label = np.load(os.path.join(args.data_path, "seizure_y.npy"))
    print('load seizure data, shape:', feature.shape, label.shape)

    # TODO: cross-validation

    # shuffle:
    shuffled_index = np.random.permutation(np.arange(feature.shape[0]))
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


def generate_tuh_data(args):
    data_path = args.data_path

    freqs = [12,24,48,64,96]
    x_data = []
    y_data = []

    if args.independent:
        # TODO
        pass
    types_dict = {}
    for freq in freqs:
        x_f_data = []
        y_f_data = []

        freq_file_name = f"fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_{freq}"
        dir, _, files = next(walk(os.path.join(data_path, freq_file_name)))
        for i, name in enumerate(files):
            fft_data = pickle.load(open(os.path.join(dir,name), 'rb'))
            if fft_data.seizure_type == 'MYSZ':
                continue
            x_f_data.append(fft_data.data)
            y_f_data.append(label_dict[fft_data.seizure_type])

        x_f_data = np.stack(x_f_data, axis=0)
        print(x_f_data.shape)
        y_f_data = np.stack(y_f_data, axis=0)
        print(y_f_data.shape)
        x_data.append(x_f_data)
        y_data.append(y_f_data)

    # check each y_f_data:
    if np.max(4 * y_data[0]- y_data[1]-y_data[2]-y_data[3]-y_data[4]) == 0:
        print('prepare save!')
        x_data = np.concatenate(x_data, axis=3)
        print('x data shape:', x_data.shape)
        np.save('seizure_x.npy', x_data)
        np.save('seizure_y.npy', y_data[0])
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


def generate_dataloader_seizure(features, labels, args):
    """
     features: [train, test, val], if val is empty then val == test
     train: B, T, N, F(12,24,48,64,96)
    """
    cates = ['train', 'test', 'val']
    datasets = dict()
    # normalize over feature dimension
    for i in range(len(features)):
        # (B, T, N, F)
        print(f'{cates[i]}, feature shape: ',features[i].shape)
        for j in range(features[i].shape[-1]):
            features[i][..., j] = normalize(features[i][..., j])

    for i in range(len(features)):
        datasets[cates[i] + '_loader'] = SeqDataLoader(features[i], labels[i], args.batch_size)

    if len(features) < 3:
        datasets['val_loader'] = SeqDataLoader(features[-1], labels[-1], args.batch_size)

    return datasets


def generate_dataloader(features, labels, args, independent, subj_id=14, shuffle=False):
    # features: (Subject, Trials, N, Seq, Channel)
    datasets = dict()
    batch_size = args.batch_size
    # if subject independent, then leave one subject as the test set, do the cross-validaton.
    # and one subject as the val.

    # if subject dependent, then mix all subject, take first 9 clips ( 9*3 = 27 trials)as training set.
    # and 6 trials from test set as val. train, val, test: 27, 12, 6

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

    # take first one session, 24 trials:
#     features = features[:, :24, ...]
#     labels = labels[:, :24]

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
        # TODO: adopt on SEED_IV
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


def rnn_train(args, datasets):
    # SummaryWriter


    import os
    dt = time.strftime("%m_%d_%H_%M", time.localtime())
    log_dir = "./tfboard/"+args.server_tag+"/" + dt
    print('tensorboard path:', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir)

    adj_mx = load_eeg_adj(args.adj_file, 'scalap')
    # randm = torch.zeros(adj_mx.shape)
    # adj_mx = torch.bernoulli(randm, p=0.9)
    # adj_mx = adj_mx.numpy()
    
    # Set corr as weighted adj:
    # random take a sample of each mood with 5 channels respectively? among every subjects
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
        adjs.append(adj_mx)

    #     model = EEGEncoder(adj_mx, args, is_gpu=args.cuda)
    if args.task == 'seizure':
        model = SEEGModel(adjs, args, is_gpu=args.cuda)
    else:
        model = EEGModel(adjs, args, is_gpu=args.cuda)
    print('model:', model)
    print('model structure:', model)
    model_dict = model.state_dict()
    if args.pretrain:
        # only load the Encoder FC:
        pre_model_dict = {k:v for k, v in torch.load(args.pre_model_path).items() if k.startswith('encoder.FC')}
        model_dict.update(pre_model_dict)
        model.load_state_dict(model_dict)
        print('pretrainok')

    print('args_cuda:', args.cuda)
    if args.cuda:
        print('rnn_train RNNBlock to cuda!')
        model.cuda()
    else:
        print('rnn_train RNNBlock to cpu!')

    # add scheduler.

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)lr_scheduler.LambdaLR
    def lr_adjust(epoch):
        if epoch < 20:
            return 1
        
        return args.lr_decay_rate ** ((epoch - 19) / 3 + 1)
    
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_adjust)
    
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    trainer = Trainer(args, model, optimizer, criterion=nn.CrossEntropyLoss())

    best_val_acc = 0
    best_unchanged_threshold = 100  # accumulated epochs of best val_mae unchanged
    best_count = 0
    best_index = -1
    train_val_metrics = []
    start_time = time.time()

    for e in range(args.epochs):
        datasets['train_loader'].shuffle()
        train_loss, train_preds = [], []

        for i, (input_data, target) in enumerate(datasets['train_loader'].get_iterator()):
            if args.cuda:
                input_data = input_data.cuda()
                target = target.cuda()

            input_data, target = Variable(input_data), Variable(target)
            loss, preds = trainer.train(input_data, target)
            # training metrics
            train_loss.append(loss)
            train_preds.append(preds)
        # validation metrics
        # TODO: pick best model with best validation evaluation.
#         datasets['val_loader'].shuffle()
        val_loss, val_preds = [], []
    
        for j, (input_data, target) in enumerate(datasets['val_loader'].get_iterator()):
            if args.cuda:
                input_data = input_data.cuda()
                target = target.cuda()
            input_data, target = Variable(input_data), Variable(target)
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
        # once got best validation model ( best_unchanged_threshold epochs unchanged), then we break.
        if m['val_acc'] > best_val_acc:
            best_val_acc = m['val_acc']
            best_count = 0
            print("update best model, epoch: ", e)
            # best_model = copy.deepcopy(trainer.model.state_dict())
            # torch.save(trainer.model.state_dict(), args.best_model_save_path)
            torch.save(trainer.model, args.best_model_save_path)

            
            print(m)
            best_index = e
        else:
            best_count += 1
        if best_count > best_unchanged_threshold:
            print('Got best')
            break

        lr_sched.step()
            
    # test metrics, save best model
    # dt = time.strftime("%m%d%H%M", time.localtime())
    # save_model_path = args.best_model_save_path+"_"+dt
    
    # load best model
    # test_model = SEEGModel(adjs, args, is_gpu=args.cuda)
    # if args.cuda:
        # test_model.cuda()
    # test_model.load_state_dict(torch.load(args.best_model_save_path))
    test_model = torch.load(args.best_model_save_path)
    trainer.model = test_model
    
    print('best_epoch:', best_index)

    test_metrics = []
    test_loss, test_preds = [], []
    for i, (input_data, target) in enumerate(datasets['test_loader'].get_iterator()):
        input_data, target = Variable(input_data), Variable(target)
        if args.cuda:
            input_data = input_data.cuda()
            target = target.cuda()
        loss, preds = trainer.eval(input_data, target)
        # add metrics
        test_loss.append(loss)
        test_preds.append(preds)
    # cal metrics as a whole:
    # reshape:
    test_preds = torch.cat(test_preds, dim=0)
    test_acc = eeg_util.calc_eeg_accuracy(test_preds, datasets['test_loader'].ys)

    m = dict(test_acc=test_acc, test_loss=np.mean(test_loss))
    m = pd.Series(m)
    print("test:")
    print(m)
    test_metrics.append(m)
    # TODO: plot confusion map:
    preds_b = test_preds.argmax(dim=1)
    
    plot_confused(preds_b, datasets['test_loader'].ys, args.fig_filename)
    plot(train_val_metrics, test_metrics, args.fig_filename)

    print('finish rnn_train!, time cost:', time.time() - start_time)
    return train_val_metrics, test_metrics

def plot_confused(preds, labels, fig_filename='confusion', index=0):

    sns.set()
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.gca()
    gts = [number_label_dict[int(l)][:-2] for l in labels]
    preds = [number_label_dict[int(l)][:-2] for l in preds]
    
    label_names = [v[:-2] for v in number_label_dict.values()]
    print(label_names)
    C2= np.around(confusion_matrix(gts, preds, labels=label_names, normalize='true'), decimals=2)

    p1 = sns.heatmap(C2, cbar=True, annot=True, ax=ax, cmap="YlGn", square=True,annot_kws={"size":9},
        yticklabels=label_names,xticklabels=label_names)

    ax.figure.savefig(fig_filename + str(index)+"_confusion.png", transparent=False, bbox_inches='tight')


def plot(train_val_metrics, test_metrics, fig_filename='mae', index=0):
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
    plt.savefig(fig_filename + str(index))
    #     plt.show()


def multi_train(args, tags="", trials=10):
    '''
    train multiple times, analyze the results, get the mean and variance.
    '''

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    test_loss = []
    test_acc = []
    #     seeds = list(np.arange(100, 200, 10))
    import os
    dt = time.strftime("%m%d%H%M", time.localtime())
    log_dir = "board_log/eeg/" + dt
    os.mkdir(log_dir)

    for i in range(trials):
        #         np.random.seed(seeds[i])
        
        features, labels = load_data(args, args.data_path, args.batch_size)
        datasets = generate_dataloader(features, labels, args, args.independent, subj_id=i)
        
        tr, te = rnn_train(args, datasets, i, log_dir=log_dir)
        test_loss.append(te[0]['test_loss'])
        test_acc.append(te[0]['test_acc'])

    # Analysis:
    test_loss_m = np.mean(test_loss)
    test_loss_v = np.std(test_loss)

    test_acc_m = np.mean(test_acc)
    test_acc_v = np.std(test_acc)

    print('%s,trials: %s, t loss mean/std: %f/%f, t acc mean/std: %f%s/%f \n' % (
        tags, trials, test_loss_m, test_loss_v, test_acc_m, '%', test_acc_v))


if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    args = eeg_util.get_common_args()
    args = args.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    #     args.data_path = './data/de_LDS'
    #     args.adj_file = './data/raw_eeg_adj.csv'
    #     args.feature_len = 5
    #     args.cuda = True

    print(args)
    #     features, labels = load_data(args, args.data_path, args.batch_size)
    #     datasets = generate_dataloader(features, labels, args, args.independent)
    if args.unit_test:
        print('Unit_test!!!!!!!!!!!!!')
        transform_SEED(args.data_path, args.batch_size)
#         adj_mx = np.ones((2, 2))
#         model = EEGModel(adj_mx, args, is_gpu=args.cuda)
#         x = torch.rand((10, 32, 512))
#         model(x)
    #         correlation_map(features)
    elif args.multi_train:
        # add a date:
        dt = time.strftime("%m-%d %H:%M", time.localtime())
        model_used = "cnn" if args.using_cnn else "gnn"
        tags = "type:" + model_used + str(dt)
        multi_train(args, tags=tags, trials=10)
    else:
        xs,ys = load_tuh_data(args)
        datasets = generate_dataloader_seizure(xs,ys,args)
        rnn_train(args, datasets)

    # t1 = time.time()
#     wavenet_train(args, datasets, scaler)
