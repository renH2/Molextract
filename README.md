# Extracting Training Data from Molecular Pre-trained Models

## About

This repo is the official code for NeurIPS'24: 'Extracting Training Data from Molecular Pre-trained Models'

## Brief Introduction 
Most of the existing methods are not suitable for data extraction in the context of graphs due to the implicit semantics present in the graph structures.

- We design a scoring function and employ an auxiliary dataset to further refine and learn the scoring function, enabling the filtration of potential training molecules.
- To efficiently extract molecules from the molecular  pre-trained model, we propose a reinforcement learning-based extraction method, utilizing the scoring function as the reward mechanism.


## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#File structure'>1. File structure</a>
* <a href='#requirements'>2. Requirements </a>
* <a href='#Usage'>3. Usage: How to run the code </a>



<span id='File structure'/>
## File Structure 

```
├── README.md
├── descriptors.py
├── docking_score
│   ├── ReLeaSE_Vina
│   │   └── docking
│   │       ├── 5ht1b
│   │       │   ├── datasets
│   │       │   │   └── 5ht1b.csv
│   │       │   ├── metadata.json
│   │       │   ├── receptor.pdbqt
│   │       │   └── receptor_copy.pdbqt
│   │       ├── fa7
│   │       │   └── receptor.pdbqt
│   │       └── parp1
│   │           └── receptor.pdbqt
│   ├── __init__.py
│   ├── bin
│   │   └── qvina02
│   ├── config_5ht1b.yaml
│   ├── config_fa7.yaml
│   ├── config_parp1.yaml
│   ├── docking_score.py
│   ├── docking_simple.py
│   ├── fpscores.pkl.gz
│   ├── sascorer.py
│   └── tmp
│       ├── dock_0.pdbqt
│       ├── ligand_0.mol
│       └── ligand_0.pdbqt
├── fcd
│   ├── ChemNet_v0.13_pretrained.pt
│   ├── __init__.py
│   ├── fcd.py
│   ├── torch_layers.py
│   └── utils.py
├── final-generate.py
├── gym_molecule
│   ├── __init__.py
│   ├── dataset
│   │   ├── descriptors_list.txt
│   │   ├── motifs_91.txt
│   │   ├── motifs_zinc_random_92.txt
│   │   ├── opt.test.logP-SA
│   │   ├── ring.txt
│   │   ├── ring_84.txt
│   │   └── scaffold_top8.txt
│   └── envs
│       ├── __init__.py
│       ├── docking_simple.py
│       ├── env_utils_graph.py
│       ├── fpscores.pkl.gz
│       ├── molecule_graph.py
│       ├── opt.test.logP-SA
│       ├── pretrained_models
│       │   ├── GNN_aux.py
│       │   ├── GNN_auxv1.py
│       │   ├── GNN_simple.py
│       │   ├── __init__.py
│       │   ├── aux_saved
│       │   │   ├── reg_stre_0.0
│       │   │   │   └── scaffold_0.pth
│       │   │   └── reg_stre_10.0
│       │   │       ├── scaffold_0.pth
│       │   │       ├── scaffold_1.pth
│       │   │       ├── scaffold_2.pth
│       │   │       ├── scaffold_3.pth
│       │   │       └── scaffold_4.pth
│       │   └── context_pred
│       │       ├── __init__.py
│       │       ├── contextpred.pth
│       │       ├── loader.py
│       │       ├── model.py
│       │       ├── tmp.ipynb
│       │       └── tmp.pkl
│       └── sascorer.py
├── loader.py
├── model.py
├── molecule_generation
│   ├── __init__.py
│   ├── dataset
│   │   ├── motifs_91.txt
│   │   └── ring_84.txt
│   ├── molecule_graph.py
│   └── utils_graph.py
├── policy_model.py
├── run_rl.py
├── sac.py
└── score.py
```

*****

Below, we will specifically explain the meaning of important file folders to help the user better understand the file structure.

`saved`: the directory for pre-trained model.

`gym_moleculars & molecule_generation`: contains the code for molecule generation

`docking_score`: contains calculation of docking score.

`fcd`: contains calculation of FCD score.

`model.py`: contains the reinforcement learning code.

`policy_model.py`: contains the code for policy model.

`run_rl.py`: contains the code for running the reinforcement learning model.

<span id='requirements'/>

## Requirements

Python module dependencies are listed in requirements.txt, which can be easily installed with pip:

`pip install -r requirements.txt`


## 3. Usage: How to run the code  <a href='#all_catelogue'>[Back to Top]</a>

To conduct extraction, you can execute `run_rl.py` as follows:

```bash
python run_rl.py \
  --gnn <pre-trained model name> \
  --scaffold <scaffold idx> \
  --name <save file name> \
  --gpu <gpu id> \
  --batch_size <batch size> \
  --bridge
```

For more detail, the help information of the main script `train_bridge.py` can be obtained by executing the following command.

```bash
python run_rl.py -h
```

**Demo:**	

```bash
python run_rl.py \
  --gnn gcl \
  --scaffold 0 \
  --name result \
  --gpu 1 \
  --batch_size 32 
```
The results will be stored in the `gen/molecule_result` directory.

## Contact
If you have any questions about the code or the paper, feel free to contact me.
Email: renh2@zju.edu.cn

## Acknowledgements
Part of this code is inspired by [Yang et al](https://github.com/AITRICS/FREED).
