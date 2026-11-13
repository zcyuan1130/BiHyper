import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from torch.cuda.amp import autocast 

from .hyper_utils import scipy_to_torch_sparse 


def _config_bool(config, key, default):
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class HyperGraphPropagation(nn.Module):
    def __init__(self, graph_blocks, device):
        super(HyperGraphPropagation, self).__init__()

        full_H = sp.block_diag([block["H"] for block in graph_blocks], format='csr')
        full_H_t = full_H.transpose().tocsr()
        d_v_inv_sqrt = np.concatenate([block["d_v_inv_sqrt"] for block in graph_blocks]).astype(np.float32)
        d_e_inv = np.concatenate([block["d_e_inv"] for block in graph_blocks]).astype(np.float32)

        self.H = scipy_to_torch_sparse(full_H, device)
        self.H_t = scipy_to_torch_sparse(full_H_t, device)
        self.register_buffer("d_v_inv_sqrt", torch.from_numpy(d_v_inv_sqrt).to(device))
        self.register_buffer("d_e_inv", torch.from_numpy(d_e_inv).to(device))

        # H = [H1 | H2] as a single concatenated incidence matrix.
        # Propagation through the full H is equivalent to H1+H2 summed
        # after separate normalization (Eq. in paper §4.3).

    def forward(self, embeddings):
        """Normalized hypergraph propagation through the full H = [H1 | H2].

        P = D_v^{-1/2} · H · D_e^{-1} · H^T · D_v^{-1/2}
        """
        x = embeddings * self.d_v_inv_sqrt.unsqueeze(1)
        x = torch.sparse.mm(self.H_t, x)
        x = x * self.d_e_inv.unsqueeze(1)
        x = torch.sparse.mm(self.H, x)
        x = x * self.d_v_inv_sqrt.unsqueeze(1)
        return x

class TailInnerConvLayer(nn.Module):
    def __init__(self, emb_dim, dropout=0.5, enable_gate=True):
        super(TailInnerConvLayer, self).__init__()
        self.enable_gate = enable_gate
        self.W_inner = nn.Linear(emb_dim, emb_dim, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.att = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid()
        ) if enable_gate else None

    def forward(self, all_emb, tail_indices, hyper_operator):
        E_tail = all_emb[tail_indices]
        W_inner_E = self.W_inner(E_tail)

        # Single propagation through the full H = [H1 | H2]
        E_prime = hyper_operator(W_inner_E)
        E_prime = self.act(self.dropout(E_prime))

        if not self.enable_gate:
            return E_prime

        W_tail = self.att(torch.cat([E_tail, E_prime], dim=1))
        enhanced_tail = E_tail * W_tail + E_prime * (1.0 - W_tail)
        return enhanced_tail

class BiHyperModule(nn.Module):
    def __init__(self, n_users, m_items, dataset, config, emb_dim, user_graph_data, item_graph_data, tail_u_idx, tail_i_idx):
        super(BiHyperModule, self).__init__()
        self.n_users = n_users
        self.m_items = m_items
        self.config = config
        self.dataset = dataset
        self.emb_dim = emb_dim
        self.tail_u_idx = tail_u_idx
        self.tail_i_idx = tail_i_idx
        self.enable_user_hypergraph = _config_bool(config, 'enable_user_hypergraph', True)
        self.enable_item_hypergraph = _config_bool(config, 'enable_item_hypergraph', True)
        self.enable_hypergraph_gate = _config_bool(config, 'enable_hypergraph_gate', True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("\n[BiHyperModule] Building factorized hypergraph operators (avoid explicit C matrix materialization)...")

        self.user_hypergraph = HyperGraphPropagation(user_graph_data, device) if self.enable_user_hypergraph else None
        self.item_hypergraph = HyperGraphPropagation(item_graph_data, device) if self.enable_item_hypergraph else None

        if self.user_hypergraph is not None:
            print(f"  - User Hypergraph H shape: {self.user_hypergraph.H.shape}")
        else:
            print("  - User Hypergraph disabled")

        if self.item_hypergraph is not None:
            print(f"  - Item Hypergraph H shape: {self.item_hypergraph.H.shape}")
        else:
            print("  - Item Hypergraph disabled")
        print(f"  - Hypergraph gate enabled: {self.enable_hypergraph_gate}")
        print("  - Hypergraph gate type: paper-original")

        self.user_gnn = TailInnerConvLayer(
            emb_dim, dropout=config['dropout'], enable_gate=self.enable_hypergraph_gate
        ) if self.enable_user_hypergraph else None
        self.item_gnn = TailInnerConvLayer(
            emb_dim, dropout=config['dropout'], enable_gate=self.enable_hypergraph_gate
        ) if self.enable_item_hypergraph else None

    def enhance_embeddings(self, base_users, base_items, weight=None):
        weight = self.config['inner_weight'] if weight is None else weight

        user_delta = torch.zeros_like(base_users)
        item_delta = torch.zeros_like(base_items)

        if abs(weight) < 1e-12:
            return base_users, base_items, user_delta, item_delta

        enhanced_users = base_users
        enhanced_items = base_items

        if self.user_hypergraph is not None and len(self.tail_u_idx) > 0:
            u_hyper_part = self.user_gnn(base_users, self.tail_u_idx, self.user_hypergraph)
            user_delta[self.tail_u_idx] = weight * u_hyper_part
            enhanced_users = base_users.clone()
            enhanced_users[self.tail_u_idx] = enhanced_users[self.tail_u_idx] + user_delta[self.tail_u_idx]

        if self.item_hypergraph is not None and len(self.tail_i_idx) > 0:
            i_hyper_part = self.item_gnn(base_items, self.tail_i_idx, self.item_hypergraph)
            item_delta[self.tail_i_idx] = weight * i_hyper_part
            if enhanced_items is base_items:
                enhanced_items = base_items.clone()
            else:
                enhanced_items = enhanced_items.clone()
            enhanced_items[self.tail_i_idx] = enhanced_items[self.tail_i_idx] + item_delta[self.tail_i_idx]

        return enhanced_users, enhanced_items, user_delta, item_delta

class BiHyperModel(nn.Module):
    def __init__(self, base_model, bihyper_module, dataset, config):
        super(BiHyperModel, self).__init__()
        self.base_model = base_model
        self.bihyper_module = bihyper_module
        self.dataset = dataset
        self.config = config
        
    def computer(self, perturbed=False, inner_weight=None):
        """Return BiHyper-enhanced embeddings and per-node enhancement deltas.

        When the base model supports perturbed views (e.g. SimGCL), this lets us
        build contrastive views in the same enhanced representation space used by
        the ranking loss.

        Returns:
            enhanced_users, enhanced_items, user_delta, item_delta
            where enhanced = base + delta for tail nodes, and delta = 0 for head nodes.
        """
        if perturbed:
            try:
                base_users, base_items = self.base_model.forward(perturbed=True)
            except TypeError:
                base_users, base_items = self.base_model.forward()
        else:
            base_users, base_items = self.base_model.forward()
        enhanced_users, enhanced_items, user_delta, item_delta = self.bihyper_module.enhance_embeddings(
            base_users, base_items, weight=inner_weight
        )
        return enhanced_users, enhanced_items, user_delta, item_delta
        
    def getUsersRating(self, users):
        all_users, all_items, _, _ = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def bpr_loss(self, users, pos, neg, inner_weight=None, lambda_c=None):
        with autocast(enabled=False):
            users_emb, items_emb, user_delta, item_delta = self.computer(inner_weight=inner_weight)
            users_emb = users_emb.float()
            items_emb = items_emb.float()
            user_delta = user_delta.float()
            item_delta = item_delta.float()

        u_emb = users_emb[users]
        pos_emb = items_emb[pos]
        neg_emb = items_emb[neg]

        u_delta = user_delta[users]
        pos_delta = item_delta[pos]

        if hasattr(self.base_model, 'get_user_embedding'):
            u_emb_eg = self.base_model.get_user_embedding(users)
            pos_emb_eg = self.base_model.get_item_embedding(pos)
            neg_emb_eg = self.base_model.get_item_embedding(neg)
        elif hasattr(self.base_model, 'user_embedding'):
            u_emb_eg = self.base_model.user_embedding(users)
            pos_emb_eg = self.base_model.item_embedding(pos)
            neg_emb_eg = self.base_model.item_embedding(neg)
        else:
            u_emb_eg = self.base_model.embedding_user(users)
            pos_emb_eg = self.base_model.embedding_item(pos)
            neg_emb_eg = self.base_model.embedding_item(neg)

        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        loss_bpr = -torch.mean(nn.functional.logsigmoid(pos_scores - neg_scores))

        reg_loss = (u_emb_eg.pow(2).sum() + pos_emb_eg.pow(2).sum() + neg_emb_eg.pow(2).sum()) / 2
        reg_loss = reg_loss / float(len(users))

        total_loss = loss_bpr + self.config['decay'] * reg_loss

        lambda_c = 0.0 if lambda_c is None else float(lambda_c)

        # Conflict loss: penalise opposing score contributions from the two hypergraph branches.
        if lambda_c > 0:
            base_u = u_emb - u_delta
            base_pos_i = pos_emb - pos_delta

            s_u = torch.sum(u_delta * base_pos_i, dim=-1)   # Δe_u^⊤ e_i
            s_i = torch.sum(base_u * pos_delta, dim=-1)     # e_u^⊤ Δe_i

            loss_conflict = torch.relu(-s_u * s_i).mean()
            total_loss = total_loss + lambda_c * loss_conflict
        else:
            loss_conflict = None

        return total_loss, loss_bpr
