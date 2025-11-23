import torch
import torch.nn.functional as F


def calc_hamming_dist(B1, B2):
    """
    计算汉明距离 (Vectorized)
    B1: [M, nbit] {-1, 1}
    B2: [N, nbit] {-1, 1}
    Returns: [M, N]
    """
    q = B2.shape[1]
    # 假设输入已经是 -1/1
    # dist = 0.5 * (nbit - B1 @ B2^T)
    distH = 0.5 * (q - B1 @ B2.t())
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    """
    计算 mAP@k (完全向量化 GPU 版本)

    Args:
        qB: 查询哈希码 [num_query, bit_len] {-1, 1}
        rB: 检索库哈希码 [num_database, bit_len] {-1, 1}
        query_L: 查询标签 [num_query, label_dim] {0, 1}
        retrieval_L: 检索库标签 [num_database, label_dim] {0, 1}
        k: Top-k 截断值 (例如 1000)，None 表示计算全量 mAP

    Returns:
        map: 平均精度均值 (scalar)
    """
    num_query = query_L.shape[0]

    # 1. 计算汉明距离 [num_query, num_database]
    dist = calc_hamming_dist(qB, rB)

    # 2. 排序 (按距离从小到大) [num_query, num_database]
    # argsort 在大数据集上可能显存较高，但在 batch 评估或常规数据集(NUS-WIDE)上通常可行
    # 如果显存爆了，可以在外部做 batch query
    _, sort_indices = torch.sort(dist, dim=1)

    # 如果指定了 k，只取前 k 个检索结果
    if k is not None:
        sort_indices = sort_indices[:, :k]

    # 3. 生成排序后的 Ground Truth 矩阵 [num_query, k]
    # 这一步通过矩阵乘法确定是否相关: (query_L @ retrieval_L.T) > 0
    # 为了节省显存，我们不生成完整的 [Q, DB] 关联矩阵，而是分块或者利用 gather
    # 鉴于通常 DB < 200k，直接 gather 效率最高

    # 注意：retrieval_L 是 [DB, LabelDim]，我们需要重排它
    # sorted_retrieval_L: [num_query, k, LabelDim]
    # 但这样显存消耗是 Q * K * L_dim。
    # 更省显存的方法：
    # gnd[i, j] = 1 if query[i] and sorted_db[i, j] share label

    # 优化显存方案：
    # 先计算关联性 (这是最耗显存的一步，如果 OOM 需要切分 Query)
    # 这里假设显存足够 (如 A100/3090/4090 跑 COCO/Flickr 没问题)
    # 对于 NUS-WIDE，建议分批调用此函数

    gnd_matrix = (query_L.mm(retrieval_L.t()) > 0).float()  # [Q, DB]

    # 根据排序索引 gather 关联性
    gnd_sorted = torch.gather(gnd_matrix, 1, sort_indices)  # [Q, k]

    # 4. 计算 AP
    # 累加相关数量 (TP count)
    cumsum = torch.cumsum(gnd_sorted, dim=1)  # [Q, k]

    # 生成位置索引 [1, 2, ..., k]
    pos = torch.arange(1, gnd_sorted.shape[1] + 1, device=gnd_sorted.device).float().view(1, -1)

    # Precision @ i = (Relevant_count_at_i) / i
    precision = cumsum / pos  # [Q, k]

    # 只有相关位置的 precision 才计入 AP
    # AP = sum(precision * is_relevant) / min(total_relevant, k)
    ap_sum = (precision * gnd_sorted).sum(dim=1)  # [Q]

    # 计算每个 Query 的总相关样本数
    total_rel = gnd_matrix.sum(dim=1)  # [Q]

    # 确定分母
    if k is not None:
        # mAP@k 的标准定义：分母通常是 min(Total_Rel, k)
        # 这样确保如果相关文档少于 k 个，全部找出来就是满分
        limit = torch.tensor(k, device=gnd_sorted.device).float()
        divisor = torch.min(total_rel, limit)
    else:
        divisor = total_rel

    # 避免除以 0
    divisor = torch.clamp(divisor, min=1e-6)

    ap = ap_sum / divisor

    return ap.mean()


# 兼容旧接口
calculate_map = calc_map_k