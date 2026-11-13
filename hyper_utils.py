import torch
import numpy as np
import scipy.sparse as sp
import gc

def get_tail_indices(dataset, percentile=80):
    """
    [新增] 适配 RecBole 的长尾节点划分函数
    根据节点的交互度 (Degree) 划分长尾节点。
    """
    # 1. 获取 RecBole 底层的交互矩阵 (COO 格式)
    inter_matrix = dataset.inter_matrix(form='coo')
    
    # 2. 计算每个 user 和 item 的度 (交互次数)
    # 注意：RecBole 内部 ID 从 1 开始，0 是 padding，所以长度是 n_users, n_items
    user_degrees = np.array(inter_matrix.sum(axis=1)).flatten()
    item_degrees = np.array(inter_matrix.sum(axis=0)).flatten()
    
    # 3. 计算分位数阈值 (排除掉 0 度的 padding 节点)
    valid_user_degrees = user_degrees[user_degrees > 0]
    valid_item_degrees = item_degrees[item_degrees > 0]
    
    u_threshold = np.percentile(valid_user_degrees, percentile)
    i_threshold = np.percentile(valid_item_degrees, percentile)
    
    print(f"\n[Tail Split] 正在计算长尾分布 (Percentile: {percentile}%)")
    print(f"  - User 阈值 (<=该交互数被视为长尾): {u_threshold:.2f}")
    print(f"  - Item 阈值 (<=该交互数被视为长尾): {i_threshold:.2f}")
    
    # 4. 提取长尾节点的内部索引 (大于 0 且小于等于阈值)
    tail_u_idx = np.where((user_degrees > 0) & (user_degrees <= u_threshold))[0]
    tail_i_idx = np.where((item_degrees > 0) & (item_degrees <= i_threshold))[0]
    
    print(f"  - 长尾 User 数量: {len(tail_u_idx)} / {dataset.user_num} (占比: {(len(tail_u_idx)/dataset.user_num)*100:.2f}%)")
    print(f"  - 长尾 Item 数量: {len(tail_i_idx)} / {dataset.item_num} (占比: {(len(tail_i_idx)/dataset.item_num)*100:.2f}%)")
    
    return tail_u_idx, tail_i_idx


def scipy_to_torch_sparse(mat, device):
    """
    [核心提速] 将 SciPy 稀疏矩阵极速转换为 PyTorch CSR 稀疏张量，避免内存多余拷贝
    """
    coo = mat.tocoo()
    
    # 使用 from_numpy 零拷贝底层数据，速度更快
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = torch.Size(coo.shape)
    
    # 1. 创建 COO 张量 -> 2. 合并重复索引 -> 3. 送入 GPU -> 4. 终极变身 CSR
    torch_sparse = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
    
    # 转换为底层硬件加速的 CSR 格式
    if hasattr(torch_sparse, 'to_sparse_csr'):
        return torch_sparse.to_sparse_csr()
    return torch_sparse


class UserInnerGraphBuilder:
    def __init__(self, n_users, n_items, tail_user_indices, tail_item_indices, enable_h2=True):
        self.n_users = n_users
        self.n_items = n_items
        self.enable_h2 = enable_h2

        # 尾部用户映射
        self.tail_u_indices = np.array(tail_user_indices, dtype=np.int64)
        self.n_tail_u = len(self.tail_u_indices)
        self.user_map = -np.ones(n_users, dtype=np.int64)
        self.user_map[self.tail_u_indices] = np.arange(self.n_tail_u)

        # 尾部物品映射
        self.tail_i_indices = np.array(tail_item_indices, dtype=np.int64)
        self.n_tail_i = len(self.tail_i_indices)
        self.item_map = -np.ones(n_items, dtype=np.int64)
        self.item_map[self.tail_i_indices] = np.arange(self.n_tail_i)

    def build_sparse_matrix(self, edge_index):
        """CPU 构建二阶超图，返回 CPU 上分块所需的数据，不直接转 GPU"""
        edge_index_cpu = edge_index.cpu().numpy()
        user_idx, item_idx = edge_index_cpu[0], edge_index_cpu[1]

        # 映射到尾部索引
        local_user_idx = self.user_map[user_idx]
        local_item_idx = self.item_map[item_idx]

        mask = (local_user_idx >= 0) & (local_item_idx >= 0)
        valid_user = local_user_idx[mask]
        valid_item = local_item_idx[mask]

        # H_1 构建
        values = np.ones(len(valid_user), dtype=np.float32)
        H_1 = sp.coo_matrix((values, (valid_user, valid_item)),
                            shape=(self.n_tail_u, self.n_tail_i)).tocsr()

        if self.enable_h2:
            # [核心优化] 二阶超图 H_2，严防内存爆炸
            # 显式使用 tocsr() 确保高效乘法
            R_T_R = (H_1.T @ H_1).tocsr() 
            R_T_R.data = np.clip(R_T_R.data, a_min=None, a_max=1.0)
            
            H_2 = (H_1 @ R_T_R).tocsr()

            # [核心优化] 强制要求 hstack 输出为 csr 格式
            H = sp.hstack([H_1, H_2], format='csr')

            del R_T_R
            del H_2
        else:
            H = H_1

        del H_1
        gc.collect()

        # 度向量
        d_v = np.array(H.sum(axis=1)).flatten()
        d_e = np.array(H.sum(axis=0)).flatten()
        direct_degree = np.array(H[:, :self.n_tail_i].sum(axis=1)).flatten()
        
        # 防止除零警告
        d_v_inv_sqrt = np.power(np.clip(d_v, 1e-8, None), -0.5)
        d_e_inv = np.power(np.clip(d_e, 1e-8, None), -1.0)

        return {
            "H": H,
            "d_e_inv": d_e_inv,
            "d_v_inv_sqrt": d_v_inv_sqrt,
            "direct_degree": direct_degree,
            "n_h1_edges": self.n_tail_i,
            "n_tail_i": self.n_tail_i,
        }


class ItemInnerGraphBuilder:
    def __init__(self, n_users, n_items, tail_user_indices, tail_item_indices, enable_h2=True):
        self.n_users = n_users
        self.n_items = n_items
        self.enable_h2 = enable_h2

        self.tail_u_indices = np.array(tail_user_indices, dtype=np.int64)
        self.n_tail_u = len(self.tail_u_indices)
        self.user_map = -np.ones(n_users, dtype=np.int64)
        self.user_map[self.tail_u_indices] = np.arange(self.n_tail_u)

        self.tail_i_indices = np.array(tail_item_indices, dtype=np.int64)
        self.n_tail_i = len(self.tail_i_indices)
        self.item_map = -np.ones(n_items, dtype=np.int64)
        self.item_map[self.tail_i_indices] = np.arange(self.n_tail_i)

    def build_sparse_matrix(self, edge_index):
        edge_index_cpu = edge_index.cpu().numpy()
        user_idx, item_idx = edge_index_cpu[0], edge_index_cpu[1]

        local_user_idx = self.user_map[user_idx]
        local_item_idx = self.item_map[item_idx]

        mask = (local_user_idx >= 0) & (local_item_idx >= 0)
        valid_user = local_user_idx[mask]
        valid_item = local_item_idx[mask]

        # 注意：这里 Item Builder 的形状反过来了 (n_tail_i, n_tail_u)
        values = np.ones(len(valid_item), dtype=np.float32)
        H_1 = sp.coo_matrix((values, (valid_item, valid_user)),
                            shape=(self.n_tail_i, self.n_tail_u)).tocsr()

        if self.enable_h2:
            # [核心优化] 二阶超图 H_2
            R_T_R = (H_1.T @ H_1).tocsr()
            R_T_R.data = np.clip(R_T_R.data, a_min=None, a_max=1.0)
            
            H_2 = (H_1 @ R_T_R).tocsr()

            # [核心优化]
            H = sp.hstack([H_1, H_2], format='csr')

            del R_T_R
            del H_2
        else:
            H = H_1

        del H_1
        gc.collect()

        d_v = np.array(H.sum(axis=1)).flatten()
        d_e = np.array(H.sum(axis=0)).flatten()
        direct_degree = np.array(H[:, :self.n_tail_u].sum(axis=1)).flatten()
        
        d_v_inv_sqrt = np.power(np.clip(d_v, 1e-8, None), -0.5)
        d_e_inv = np.power(np.clip(d_e, 1e-8, None), -1.0)

        return {
            "H": H,
            "d_e_inv": d_e_inv,
            "d_v_inv_sqrt": d_v_inv_sqrt,
            "direct_degree": direct_degree,
            "n_h1_edges": self.n_tail_u,
            "n_tail_i": self.n_tail_i,
        }
