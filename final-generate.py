import os
import sys

sys.path.insert(0, os.getcwd())
from multiprocessing import Pool, Manager, Process
import torch
import pickle as pkl
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.QED import qed
from model import GNN, GNN_graphpred
import argparse
from molecule_generation.utils_graph import *
from molecule_generation.molecule_graph import Molecule_Graph
import docking_score.sascorer as sascorer
from docking_score.docking_score import docking_sc


def process_records_freq(records, mol_bank, mol_bank_aux, return_records):
    for record in tqdm(records):
        freq = record.potential_subgraph_mg.get_freq(mol_bank)
        record.other['freq'] = freq
        freq_aux = record.potential_subgraph_mg.get_freq(mol_bank_aux)
        record.other['freq_aux'] = freq_aux

    return_records.extend(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("mol_generation")
    parser.add_argument('--scaffold_idx', default=0, type=int)
    parser.add_argument('--cpu_num', default=100, type=int)
    parser.add_argument('--model_type', type=str, default='gcl')
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--savefile', type=str, default='gcl')
    args = parser.parse_args()

    scaffold_smi = SCAFFOLD_VOCAB[args.scaffold_idx]

    mg = Molecule_Graph(scaffold_smi)
    try:
        core2id = pkl.load(open(r'./data_preprocessing/zinc/core2mol_bank/core2id.pkl', 'rb'))
        mol_bank_id = core2id[Chem.MolToSmiles(mg.mol_core)]
        mol_bank_filtered = pkl.load(
            open(f'./data_preprocessing/zinc/core2mol_bank/{mol_bank_id}.pkl', 'rb'))
    except:
        mol_bank = pkl.load(open(r'/data/renhong/mol-hrh/dataset/zinc_standard_agent/processed/mols.pkl', 'rb'))
        mol_core = mg.mol_core
        mol_bank_filtered = [mol for mol in tqdm(mol_bank) if mol.HasSubstructMatch(mol_core)]

    mol_core = mg.mol_core
    mol_bank_aux = pkl.load(
        open('/data/renhong/mol-attack/dataset/zinc_standard_agent_aux_20k/processed/mols.pkl', 'rb'))
    mol_bank_aux_filtered = [mol for mol in tqdm(mol_bank_aux) if mol.HasSubstructMatch(mol_core)]

    attach_points = mg.attach_points
    motif_bank_size = len(FRAG_VOCAB)

    records = []
    for p in tqdm(range(len(mg.attach_points))):
        for i in range(motif_bank_size):
            for np in range(len(FRAG_VOCAB_ATT[i])):
                potential_subgraph = copy.deepcopy(mg)
                ac = [p, i, np]
                potential_subgraph.add_motif(ac)
                records.append(record(scaffold_mg=copy.deepcopy(mg), side_chain_mg=Molecule_Graph(FRAG_VOCAB[i][0]),
                                      potential_subgraph_mg=potential_subgraph, scaffold_att_pos=p,
                                      side_chain_idx=i, side_chain_att_pos=np, other=dict()))

    pool = Pool(args.cpu_num)

    manager = Manager()
    return_records = manager.list()
    print("----------")

    size = int(len(records) / args.cpu_num) + 1
    for idx, i in enumerate(range(0, len(records), size)):
        pool.apply_async(func=process_records_freq,
                         args=(records[i:i + size], mol_bank_filtered, mol_bank_aux_filtered, return_records))

    pool.close()
    pool.join()

    records = [r for r in return_records]

    records_new = []
    for r in records:
        r = record(scaffold_mg=r.scaffold_mg,side_chain_mg=Molecule_Graph(FRAG_VOCAB[r.side_chain_idx][0]),
                   potential_subgraph_mg=r.potential_subgraph_mg,scaffold_att_pos=r.scaffold_att_pos,
                   side_chain_idx=r.side_chain_idx,side_chain_att_pos=r.side_chain_att_pos,other=r.other)
        records_new.append(r)
    records = records_new

    for r in tqdm(records):
        r.other['qed'] = qed(r.potential_subgraph_mg.mol)
        r.other['sa'] = -1 * sascorer.calculateScore(r.potential_subgraph_mg.mol)

    smiles = [Chem.MolToSmiles(r.potential_subgraph_mg.mol_core) for r in records]
    for protein in ['fa7', 'parp1', '5ht1b']:
        scores = docking_sc([Chem.MolToSmiles(r.potential_subgraph_mg.mol_core) for r in records], protein)
        for r, s in zip(records, scores):
            r.other[f"docking_{protein}"] = -1 * s

    device = args.device if args.device is not None else 'cpu'
    model = GNN_graphpred(5, 300, 1, JK='last', drop_ratio=0.5, graph_pooling='mean').to(device)
    if args.model_type == 'contextpred':
        state = torch.load('./saved/contextpred.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'gcl':
        state = torch.load('./saved/graphcl_80.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'masking':
        state = torch.load('./saved/masking.pth')
    elif args.model_type == 'infomax':
        state = torch.load('./saved/infomax.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'mae':
        state = torch.load('./saved/pretrained.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'simgrace':
        state = torch.load('./saved/simgrace_100.pth')
        model.gnn.load_state_dict(state)
    elif args.model_type == 'molebert':
        state = torch.load('./saved/Mole-BERT.pth')
        model.gnn.load_state_dict(state)
        model.gnn.load_state_dict(state)
    elif args.model_type == 'supervised':
        state = torch.load('./saved/supervised.pth')
        model.gnn.load_state_dict(state)
    else:
        raise NotImplementedError
    model.eval()


    for r in tqdm(records):
        M = r.scaffold_mg.atom_num
        N = r.side_chain_mg.atom_num
        representation_M = r.scaffold_mg.representation(model, device)
        representation_N = r.side_chain_mg.representation(model, device)
        representation_total = r.potential_subgraph_mg.representation(model, device)
        r.other['representation_scaffold'] = representation_M.detach().cpu().squeeze().numpy()
        r.other['representation_side_chain'] = representation_N.detach().cpu().squeeze().numpy()
        r.other['representation_merged_mol'] = representation_total.detach().cpu().squeeze().numpy()

    if not os.path.exists(f'./records/records_{args.model_type}_ring'):
        os.makedirs(f'./records/records_{args.model_type}_ring')
    pkl.dump(records,
             open(f'./records/records_{args.model_type}_ring/scaffold_{args.scaffold_idx}.pkl','wb'))
