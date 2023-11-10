# Graph‐generative neural network for EEG‐based epileptic seizure detection via discovery of dynamic brain functional connectivity
- [paper link](https://www.nature.com/articles/s41598-022-23656-1)

GGN is a generative deep learning model for epilepsy seizure classification and detecting the abnormal functional connectivities when seizure attacks.

If any code or the datasets are useful in your research, please cite the following paper:

```
@article{li2022graph,
  title={Graph-generative neural network for EEG-based epileptic seizure detection via discovery of dynamic brain functional connectivity},
  author={Li, Zhengdao and Hwang, Kai and Li, Keqin and Wu, Jie and Ji, Tongkai},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={18998},
  year={2022},
  publisher={Nature Publishing Group UK London}
}


```

or 

```
Li, Z., Hwang, K., Li, K. et al. Graph-generative neural network for EEG-based epileptic seizure detection via discovery of dynamic brain functional connectivity. Sci Rep 12, 18998 (2022).
```

--- 

## Preparation for dataset. 


1. According to the Policy from TUH, you must apply dataset TUSZ v1.5.2 from https://isip.piconepress.com/projects/tuh_eeg/
2. Preprocess the raw data following the benchmark setting from IBM: https://github.com/IBM/seizure-type-classification-tuh, after that, you get fft_seizures_wl1_ws_0.25_sf_250_fft_min_1_fft_max.... files. 
3. Composite features from different frequencies following our paper (Supplementary). We provide a funtion to generate features, you could set: args.task == 'generate_data' in the training.sh file, or specify when you train: `sh training.sh --task=generate_data`, for more details of feature generation, please check the function: `generate_tuh_data` in eeg_main.py file where there are some hyperparameters in it.

Details refer to the [suplementary](https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-022-23656-1/MediaObjects/41598_2022_23656_MOESM1_ESM.pdf)

---


## Testing via trained model.

the shuffled_index.npy stored the indices of training samples and testing samples of the best_models/ggn_best.pth (reported in the paper).

1. config the data path and trained model path in the file `testing.sh`.
1. `sh testing.sh`
1. use `sh testing.sh kill`, to kill the running process.

## Training GGN

1. config the data path and trained model path in the file training.sh
2. `sh training.sh`
3. or you could reset the hyperparameters in training.sh or just set in args, e.g.,`
sh training.sh data_path=xxx lr=0.00005`
1. use `sh training.sh kill`, to kill the running process.

## Training compared models

To train compared models, chanage the `--task=ggn` to following settings:


1. `sh training.sh --task=cnnnet`, training CNN based model.
1. `sh training.sh --task=gnnnet`, training GNN based model.
1. `sh training.sh --task=transformer`, training Transformer based model.

## Training logs

**Note that**, we use `nohup` to run the program in background, the log path is specified in `training.sh`.

## Turn on debug log mode
to print more logs, set `--debug` in the command args.

## Q&A

**Message me if you have any questions about code or data: zhengdaoli (at) link (dot) cuhk (dot) edu.cn**
