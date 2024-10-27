from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from molecule_generation.utils_graph import *


class LearnAlphaNew(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.dim = 100
        self.transform_motif = nn.Sequential(nn.Linear(emb_dim, self.dim), )
        self.transform_scaffold = nn.Sequential(nn.Linear(emb_dim, self.dim), )
        self.transform_merged = nn.Sequential(nn.Linear(emb_dim, self.dim), )
        self.transform_alpha = nn.Sequential(nn.Linear(3 * emb_dim, 1), )

    def forward(self, motif_representations, scaffold_representations, merged_mol_representations, scaffold_sizes,
                motif_sizes):
        x = torch.cat([scaffold_representations, motif_representations, merged_mol_representations], dim=-1)
        alpha = self.transform_alpha(x)
        side_chain_representations = self.transform_motif(motif_representations)
        scaffold_representations = self.transform_scaffold(scaffold_representations)
        merged_mol_representations = self.transform_merged(merged_mol_representations)
        estimated_representations = (alpha * scaffold_representations * scaffold_sizes + (
                1 - alpha) * side_chain_representations * motif_sizes) / (scaffold_sizes + motif_sizes)
        score = F.cosine_similarity(estimated_representations, merged_mol_representations)
        return alpha.cpu().detach().numpy(), score.cpu().detach().numpy(), estimated_representations.cpu().detach().numpy(), side_chain_representations.cpu().detach().numpy(), scaffold_representations.cpu().detach().numpy(), merged_mol_representations.cpu().detach().numpy()

    def loss(self, labels, motif_representations, scaffold_representations, merged_mol_representations,
             scaffold_sizes, motif_sizes):
        x = torch.cat([scaffold_representations, motif_representations, merged_mol_representations], dim=-1)
        alpha = self.transform_alpha(x)

        motif_representations = self.transform_motif(motif_representations)
        scaffold_representations = self.transform_scaffold(scaffold_representations)
        merged_mol_representations = self.transform_merged(merged_mol_representations)
        estimated_representations = (alpha * scaffold_representations * scaffold_sizes + (
                1 - alpha) * motif_representations * motif_sizes) / (scaffold_sizes + motif_sizes)
        score1 = F.cosine_similarity(estimated_representations, merged_mol_representations)
        weight = labels.sum() / labels.size(0)

        loss = torch.where(labels >= 1.0, -1 * score1, weight * score1).sum()
        loss_alpha = torch.where((0.0 < alpha) & (alpha <= 1.0), torch.tensor([0.0]), torch.abs(alpha)).sum()
        loss = loss + 50 * loss_alpha
        return loss


def get_representations(records):
    scaffold_representations = []
    side_chain_representations = []
    merged_mol_representations = []
    scaffold_sizes = []
    side_chain_sizes = []

    labels = []
    labels_ground = []

    invalid_count = 0
    for r in tqdm(records):
        if np.abs(r.other['representation_merged_mol']).max() > 1e4:
            invalid_count += 1
            continue
        scaffold_representations.append(r.other['representation_scaffold'])
        side_chain_representations.append(r.other['representation_side_chain'])
        merged_mol_representations.append(r.other['representation_merged_mol'])
        scaffold_sizes.append(r.scaffold_mg.atom_num)
        side_chain_sizes.append(r.side_chain_mg.atom_num)

        labels.append(1. if r.other['freq_aux'] > 0 else 0.)
        labels_ground.append(1. if r.other['freq'] > 0 else 0.)

    scaffold_representations = torch.tensor(scaffold_representations)
    scaffold_sizes = torch.tensor(scaffold_sizes)[:, np.newaxis]
    side_chain_representations = torch.tensor(side_chain_representations)
    side_chain_sizes = torch.tensor(side_chain_sizes)[:, np.newaxis]
    merged_mol_representations = torch.tensor(merged_mol_representations)
    labels = torch.tensor(labels)
    labels_ground = torch.tensor(labels_ground)

    return scaffold_representations, scaffold_sizes, side_chain_representations, side_chain_sizes, merged_mol_representations, labels, labels_ground

