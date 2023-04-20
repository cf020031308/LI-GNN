# Label-inputted Implicit Graph Neural Network (LI-GNN)

This repo contains code that are required to reporduce all experiments in our paper *Graph Equilibrium Networks: Unifying Label-inputted Graph Neural Networks with Deep Equilibrium Models*.

## Paper Abstract

The success of Graph Neural Networks (GNN) in learning on non-Euclidean data arouses many subtopics, such as Label-inputted GNN (LGNN) and Implicit GNN (IGNN).
LGNN, explicitly inputting supervising information in GNN, integrates label propagation to achieve superior performance, but with the dilemma between its propagating distance and adaptiveness.
IGNN, outputting an equilibrium point by iterating its network infinite times, exploits information in the entire graph to capture long-range dependencies, but with its network constrained to guarantee the existence of the equilibrium.
This work unifies the two subdomains by interpreting LGNN in the theory of IGNN and reducing prevailing LGNNs to forms of IGNN.
The unification facilitates the exchange between the two subdomains and inspires more studies.
Specifically, implicit differentiation of IGNN is introduced to LGNN to differentiate its infinite-range label propagation with constant memory, making the propagation both distant and adaptive.
Besides, the masked label strategy of LGNN is proven able to guarantee the well-posedness of IGNN in a network-agnostic manner, granting its network more complex and thus more expressive.
Combining the advantages of LGNN and IGNN, Label-inputted Implicit GNN is proposed.
Node classification experiments on two synthesized and seven real-world datasets demonstrate its effectiveness.

## Usage

```
usage: mai2.pyc [-h] [--runs RUNS] [--gpu GPU] [--split SPLIT] [--lr LR] [--dropout DROPOUT] [--n-layers N_LAYERS] [--weight-decay WEIGHT_DECAY]
                [--early-stop-epochs EARLY_STOP_EPOCHS] [--max-epochs MAX_EPOCHS] [--hidden HIDDEN] [--heads HEADS] [--alpha ALPHA] [--theta THETA]
                [--correct CORRECT] [--correct-rate CORRECT_RATE] [--smooth SMOOTH] [--smooth-rate SMOOTH_RATE] [--input-label INPUT_LABEL]
                [--for-iter FOR_ITER] [--back-iter BACK_ITER] [--drop-state DROP_STATE] [--inductive]
                method dataset

positional arguments:
  method                MLP | SGC | GCN | IGNN | EIGNN | GIN | SAGE | GAT | GCNII | JKNet
  dataset               cora | citeseer | pubmed | flickr | arxiv | yelp | reddit | ...

options:
  -h, --help            show this help message and exit
  --runs RUNS           Default: 1
  --gpu GPU             Default: 0
  --split SPLIT         Ratio of labels for training. Set to 0 to use default split (if any) or 0.6. With an integer x the dataset is splitted like
                        Cora with the training set be composed by x samples per class. Default: 0
  --lr LR               Learning Rate. Default: 0.01
  --dropout DROPOUT     Default: 0
  --n-layers N_LAYERS   Default: 2
  --weight-decay WEIGHT_DECAY
                        Default: 0
  --early-stop-epochs EARLY_STOP_EPOCHS
                        Maximum epochs until stop when accuracy decreasing. Default: 100
  --max-epochs MAX_EPOCHS
                        Maximum epochs. Default: 1000
  --hidden HIDDEN       Dimension of hidden representations and implicit state. Default: 64
  --heads HEADS         Number of attention heads for GAT. Default: 0
  --alpha ALPHA         Hyperparameter for GCNII. Default: 0.5
  --theta THETA         Hyperparameter for GCNII. Default: 1.0
  --correct CORRECT     Iterations for Correct after prediction. Default: 0
  --correct-rate CORRECT_RATE
                        Propagation rate for Correct after prediction. Default: 0.1
  --smooth SMOOTH       Iterations for Smooth after prediction. Default: 0
  --smooth-rate SMOOTH_RATE
                        Propagation rate for Smooth after prediction. Default: 0.1
  --input-label INPUT_LABEL
                        Ratio of known labels for input. Default: 0
  --for-iter FOR_ITER   Iterations to produce state in forward-pass. Default: 0
  --back-iter BACK_ITER
                        Iterations to accumulate vjp in backward-pass. Default: 0
  --drop-state DROP_STATE
                        Ratio of state for dropping. Default: 0
  --inductive           Enable the inductive setting
```

For example, if you want to run MLP on the Cora dataset with its default split on gpu `cuda:3` for 5 runs, execute

```bash
python3 main.py MLP cora --split 0 --gpu 3 --runs 5
```

## Reproducibility

Files in `scripts/` folder are scripts that reproduce experiments in our article.

* `run_baseline` runs experiments to produces accuracy scores for IGNN and SotA in Table 2
* `run_chain` runs experiment on the Chains dataset, producing data for Figure 4.
* `run_gcn` runs experiments to produce accuracy scores for GCN, C&S, Label Tricks, and LI-GNN in Table 2
* `run_sota` runs experiments to produce accuracy scores for C&S, Label Tricks, and LI-GNN with SotA as their backbones in Table 2
* `run_weekday` runs experiment on the Weekday dataset, producing data for Figure 3.
* `run_wellpose` runs experiments to produce data for Figure 2.

## Datasets

The Chain datasets are generated by code slightly modified from [IGNN](https://github.com/SwiftieH/IGNN).
The Weekday dataset is generated by code from [3ference](https://github.com/cf020031308/3ference).
Other datasets used in our paper, including Chameleon, Squirrel, Flickr, Pubmed, Amazon Photo, Coauthor CS, and etc, are retrieved with [DGL](https://github.com/dmlc/dgl) and [PyG](https://github.com/pyg-team/pytorch_geometric).

Datasets that did not appear in our paper can also be retrieved by our code for further exploration, with the help of DGL, PyG, and [OGB](https://github.com/snap-stanford/ogb).

## Baselines

We implement all methods except IGNN and EIGNN with PyTorch, PyG, and Scikit-learn, if you want to run one of them, install PyTorch, Scikit-learn, DGL, PyG, and OGB first then execute `main.py`.

To run IGNN and EIGNN, you need to clone [this commit of EIGNN](https://github.com/liu-jc/EIGNN/tree//6a2c8e73c11bfebc8614d955226dbae600cc8dfc) (and install its dependencies) then place our `main.py` into the cloned folder.

## Citation

```bibtex
@misc{luo2022unifying,
      title={Unifying Label-inputted Graph Neural Networks with Deep Equilibrium Models}, 
      author={Yi Luo and Guiduo Duan and Guangchun Luo and Aiguo Chen},
      year={2022},
      eprint={2211.10629},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
