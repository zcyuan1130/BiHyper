import torch
import importlib
import numpy as np
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

# 使用相对导入
from .bihyper_model import BiHyperModel, BiHyperModule
from .hyper_utils import UserInnerGraphBuilder, ItemInnerGraphBuilder


def _config_bool(config, key, default):
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _config_float(config, key, default):
    if key not in config:
        return default
    value = config[key]
    if value is None:
        return default
    return float(value)


class BiHyper(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BiHyper, self).__init__(config, dataset)
        
        self.n_users = dataset.user_num
        self.m_items = dataset.item_num
        self.emb_dim = config['embedding_size']
        self.device = config['device']
        self.inner_weight = config['inner_weight'] if 'inner_weight' in config else 1.0
        self.bihyper_warmup_steps = config['bihyper_warmup_steps'] if 'bihyper_warmup_steps' in config else 0
        self.cl_every_n_steps = config['cl_every_n_steps'] if 'cl_every_n_steps' in config else 1
        self.cl_start_step = config['cl_start_step'] if 'cl_start_step' in config else 0
        self.enable_user_hypergraph = _config_bool(config, 'enable_user_hypergraph', True)
        self.enable_item_hypergraph = _config_bool(config, 'enable_item_hypergraph', True)
        self.enable_hypergraph_h2 = _config_bool(config, 'enable_hypergraph_h2', True)
        self.enable_hypergraph_gate = _config_bool(config, 'enable_hypergraph_gate', True)
        self.lambda_c = _config_float(config, 'lambda_c', 0.0)
        self.train_step = 0
        
        # ==========================================
        # 动态加载基础模型 (例如 LightGCN, SimGCL)
        # ==========================================
        self.base_model_name = config['base_model_name'] if 'base_model_name' in config else 'LightGCN'
        
        try:
            module_path = f'recbole_gnn.model.general_recommender.{self.base_model_name.lower()}'
            module = importlib.import_module(module_path)
        except ImportError:
            module_path = f'recbole.model.general_recommender.{self.base_model_name.lower()}'
            module = importlib.import_module(module_path)
            
        BaseModelClass = getattr(module, self.base_model_name)
        self.base_model = BaseModelClass(config, dataset).to(self.device)
        print(f"\n[BiHyper Adapter] Successfully dynamically loaded Base Model: {self.base_model_name}")

        # ==========================================
        # 提取交互数据并动态划分长尾节点 (80/20法则)
        # ==========================================
        inter_matrix = dataset.inter_matrix(form='coo')
        edge_index = torch.tensor(
            np.vstack((inter_matrix.row, inter_matrix.col)), 
            dtype=torch.long
        ).to(self.device)
        
        user_degrees = np.array(inter_matrix.sum(axis=1)).flatten()
        item_degrees = np.array(inter_matrix.sum(axis=0)).flatten()
        
        tail_ratio = config['tail_ratio'] if 'tail_ratio' in config else 0.8
        valid_u_degrees = user_degrees[user_degrees > 0]
        valid_i_degrees = item_degrees[item_degrees > 0]
        
        u_threshold = np.percentile(valid_u_degrees, tail_ratio * 100)
        i_threshold = np.percentile(valid_i_degrees, tail_ratio * 100)
        
        print(f"\n[BiHyper] Dynamic Tail Splitting (Ratio: {tail_ratio*100}%):")
        print(f"  - User Degree Threshold: <= {u_threshold:.1f}")
        print(f"  - Item Degree Threshold: <= {i_threshold:.1f}\n")
        print("[BiHyper] Enabled hypergraph branches:")
        print(f"  - User-side hypergraph: {self.enable_user_hypergraph}")
        print(f"  - Item-side hypergraph: {self.enable_item_hypergraph}")
        print(f"  - Second-order hypergraph H2: {self.enable_hypergraph_h2}")
        print(f"  - Hypergraph gate: {self.enable_hypergraph_gate}")
        print(f"  - Conflict loss lambda_c: {self.lambda_c}\n")
        
        self.tail_u_idx = np.where((user_degrees > 0) & (user_degrees <= u_threshold))[0]
        self.tail_i_idx = np.where((item_degrees > 0) & (item_degrees <= i_threshold))[0]

        # ==========================================
        # 构建超图并准备 factorized hypergraph operator
        # ==========================================
        u_graph_data = None
        i_graph_data = None

        if self.enable_user_hypergraph:
            u_builder = UserInnerGraphBuilder(
                self.n_users, self.m_items, self.tail_u_idx, self.tail_i_idx,
                enable_h2=self.enable_hypergraph_h2
            )
            u_graph_data = u_builder.build_sparse_matrix(edge_index)

        if self.enable_item_hypergraph:
            i_builder = ItemInnerGraphBuilder(
                self.n_users, self.m_items, self.tail_u_idx, self.tail_i_idx,
                enable_h2=self.enable_hypergraph_h2
            )
            i_graph_data = i_builder.build_sparse_matrix(edge_index)
        
        # 实例化核心模块
        self.bihyper_module = BiHyperModule(
            self.n_users, self.m_items, dataset, config, self.emb_dim,
            [u_graph_data] if u_graph_data is not None else [],
            [i_graph_data] if i_graph_data is not None else [],
            self.tail_u_idx, self.tail_i_idx
        )
        
        self.my_bihyper = BiHyperModel(self.base_model, self.bihyper_module, dataset, config)

    def _get_train_inner_weight(self):
        if self.bihyper_warmup_steps <= 0:
            return self.inner_weight

        progress = min(1.0, self.train_step / float(self.bihyper_warmup_steps))
        return self.inner_weight * progress

    def _should_run_cl(self):
        if self.cl_every_n_steps <= 0:
            return False
        if self.train_step < self.cl_start_step:
            return False
        return (self.train_step - self.cl_start_step) % self.cl_every_n_steps == 0

    def forward(self):
        enhanced_users, enhanced_items, _, _ = self.my_bihyper.computer()
        return enhanced_users, enhanced_items

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        self.train_step += 1
        train_inner_weight = self._get_train_inner_weight()
        
        # 1. 计算 BiHyper 增强后的主任务损失 (BPR + L2 + Conflict)
        total_loss, _ = self.my_bihyper.bpr_loss(
            users, pos_items, neg_items, inner_weight=train_inner_weight, lambda_c=self.lambda_c
        )
        
        # ==========================================
        # 2. 核心：融合 SimGCL 的对比学习损失
        # ==========================================
        if self.base_model_name == 'SimGCL' and self._should_run_cl():
            # 获取去重后的用户和物品 (与 SimGCL 源码完全对齐)
            unique_users = torch.unique(users)
            unique_pos_items = torch.unique(pos_items)
            
            # CL 保持在 SimGCL 的 base 表示空间中计算，避免每个对比视图都重复执行
            # BiHyper 超图增强；BPR 仍然使用 BiHyper 增强后的 clean 表示。
            perturbed_u_embs_1, perturbed_i_embs_1 = self.base_model.forward(perturbed=True)
            perturbed_u_embs_2, perturbed_i_embs_2 = self.base_model.forward(perturbed=True)
            
            # 计算 InfoNCE 对比损失
            user_cl_loss = self.base_model.calculate_cl_loss(
                perturbed_u_embs_1[unique_users], 
                perturbed_u_embs_2[unique_users]
            )
            item_cl_loss = self.base_model.calculate_cl_loss(
                perturbed_i_embs_1[unique_pos_items], 
                perturbed_i_embs_2[unique_pos_items]
            )
            
            # 将对比损失按照 cl_rate 加到总 Loss 中
            total_loss += self.base_model.cl_rate * (user_cl_loss + item_cl_loss)

        return total_loss

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        
        all_users, all_items = self.forward()
        u_emb = all_users[users]
        i_emb = all_items[items]
        
        scores = torch.mul(u_emb, i_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        
        all_users, all_items = self.forward()
        u_emb = all_users[user]
        
        scores = torch.matmul(u_emb, all_items.transpose(0, 1))
        return scores.view(-1)
