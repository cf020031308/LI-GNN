# Graph Equilibrium Networks

This repo contains code that are required to reporduce all experiments in our paper *Graph Equilibrium Networks: Unifying Label-inputted Graph Neural Networks with Deep Equilibrium Models*.

## Paper Abstract

For node classification, Graph Neural Networks (GNN) assign predefined labels to graph nodes according to node features propagated along the graph structure. Apart from the traditional end-to-end manner inherited from deep learning, many subsequent works input assigned labels into GNNs to improve their classification performance. Such label-inputted GNNs (LGNN) combine the advantages of learnable feature propagation and long-range label propagation, producing state-of-the-art performance on various benchmarks. However, the theoretical foundations of LGNNs are not well-established, and the combination is with seam because the long-range propagation is memory-consuming for optimization. To this end, this work interprets LGNNs with the theory of Implicit GNN (IGNN), which outputs a fixed state point of iterating its network infinite times and optimizes the infinite-range propagation with constant memory consumption. Besides, previous contributions to LGNNs inspire us to overcome the heavy computation in training IGNN by iterating the network only once but starting from historical states, which are randomly masked in forward-pass to implicitly guarantee the existence and uniqueness of the fixed point. Our improvements to IGNNs are network agnostic: for the first time, they are extended with complex networks and applied to large-scale graphs. Experiments on two synthetic and six realworld datasets verify the advantages of our method in terms of long-range dependencies capturing, label transitions modelling, accuracy, scalability, efficiency, and well-posedness.

## Usage

```
usage: help.py [-h] [--hidden HIDDEN] [--runs RUNS] [--gpu GPU] [--lr LR]
               [--dropout DROPOUT] [--n-layers N_LAYERS]
               [--weight-decay WEIGHT_DECAY] [--label-input LABEL_INPUT]
               [--label-reuse LABEL_REUSE] [--split SPLIT] [--correct CORRECT]
               [--correct-rate CORRECT_RATE] [--smooth SMOOTH]
               [--smooth-rate SMOOTH_RATE] [--no-self-loops] [--asymmetric]
               [--early-stop-epochs EARLY_STOP_EPOCHS]
               [--max-epochs MAX_EPOCHS] [--skip-connection]
               [--attention ATTENTION] [--noise NOISE]
               [--drop-state DROP_STATE]
               method dataset

positional arguments:
  method                MLP | SGC | GCN | IGNN | EIGNN | GQN
  dataset               cora | citeseer | pubmed | flickr | ppi | arxiv | yelp
                        | reddit | ...

optional arguments:
  -h, --help            show this help message and exit
  --hidden HIDDEN       Dimension of hidden representations. Default: 64
  --runs RUNS           Default: 1
  --gpu GPU             Default: 0
  --lr LR               Learning Rate. Default: 0.01
  --dropout DROPOUT     Default: 0
  --n-layers N_LAYERS   Default: 2
  --weight-decay WEIGHT_DECAY
                        Default: 0
  --label-input LABEL_INPUT
                        Ratio of known labels for input. Default: 0
  --label-reuse LABEL_REUSE
                        Iterations to produce pseudo labels for label input.
                        Default: 0
  --split SPLIT         Ratio of labels for training. Set to 0 to use default
                        split (if any). Default: 0.6
  --correct CORRECT     Iterations for Correct after prediction. Default: 0
  --correct-rate CORRECT_RATE
                        Propagation rate for Correct after prediction.
                        Default: 0.1
  --smooth SMOOTH       Iterations for Smooth after prediction. Default: 0
  --smooth-rate SMOOTH_RATE
                        Propagation rate for Smooth after prediction. Default:
                        0.1
  --no-self-loops       Add self loops. Default: yes
  --asymmetric          Treat the graph as directional (if it is). Default:
                        symmetric
  --early-stop-epochs EARLY_STOP_EPOCHS
                        Maximum epochs until stop when accuracy decreasing.
                        Default: 100
  --max-epochs MAX_EPOCHS
                        Maximum epochs. Default: 1000
  --skip-connection     Enable skip connections (a.k.a. linear layer).
                        Default: disabled
  --attention ATTENTION
                        Number of attention heads. Default: 0
  --noise NOISE         Weight of standalone noise inputted for
                        regularization. Default: 0
  --drop-state DROP_STATE
                        Dropout probability for inputted state. Default: 0
```

For example, if you want to run MLP on the Cora dataset with its default split on gpu `cuda:3` for 5 runs, execute

```bash
python3 main.py MLP cora --split 0 --gpu 3 --runs 5
```

## Reproducibility

Files in `scripts/` folder are scripts that reproduce experiments in our article.

* `run_chain` runs experiment on the Chains dataset, producing data for Figure 2.
* `run_weekday` runs experiment on the Weekday dataset, producing data for Figure 2.
* `run_baseline` and `run_baseline_full` produces results in Table 2, reporting accuracy scores on the six datasets for GQN and baseline methods.
* `run_ablation` runs GQN with different regularization methods. It produces the bar chart in Appendix of the article.

## Datasets

The Chain datasets are generated by code slightly modified from [IGNN](https://github.com/SwiftieH/IGNN).
The Weekday dataset is generated by code from [3ference](https://github.com/cf020031308/3ference).
Other datasets used in our paper, including Chameleon, Squirrel, Flickr, Pubmed, Amazon Photo, and Coauthor CS, are retrieved with [DGL](https://github.com/dmlc/dgl), [PyG](https://github.com/pyg-team/pytorch_geometric).

Datasets that did not appear in our paper can also be retrieved by our code for further exploration, with the help of DGL, PyG, and [OGB](https://github.com/snap-stanford/ogb).

## Baselines

We implement all methods except IGNN and EIGNN with PyTorch and Scikit-learn, if you want to run one of them, install PyTorch, Scikit-learn, DGL, PyG, and OGB first then execute `main.py`.

To run IGNN and EIGNN, you need to clone [this commit of EIGNN](https://github.com/liu-jc/EIGNN/tree//6a2c8e73c11bfebc8614d955226dbae600cc8dfc) (and install its dependencies) then place our `main.py` into the cloned folder.

## Citation

```
```
