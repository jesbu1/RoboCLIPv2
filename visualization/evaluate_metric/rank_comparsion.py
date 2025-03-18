import numpy as np
import wandb
def rank_comparison(cm1, cm2, cm3):
    """
    只比较混淆矩阵对角线上的值 (每个任务的正确预测数)，
    给出每个矩阵在各任务上的排名(1=最好,3=最差)，
    并统计每个矩阵拿到排名1,2,3的概率，最后上传到W&B。
    """

    # 1. 取出三个矩阵的对角线
    diag1 = np.diag(cm1)
    diag2 = np.diag(cm2)
    diag3 = np.diag(cm3)

    num_tasks = len(diag1)

    # ranks[i, j] 表示第 i 个任务下，第 j 个矩阵的排名(1/2/3)
    ranks = np.zeros((num_tasks, 3), dtype=np.int32)

    # 2. 逐任务比较
    for i in range(num_tasks):
        d1, d2, d3 = diag1[i], diag2[i], diag3[i]
        values = [d1, d2, d3]
        
        # 从小到大排序，返回索引
        sorted_indices = np.argsort(values)  # ascending order
        
        # sorted_indices[0] -> 最小值, sorted_indices[2] -> 最大值
        # 我们要给最大值排名 1，最小值排名 3
        ranks[i, sorted_indices[0]] = 3  # 最小值
        ranks[i, sorted_indices[1]] = 2
        ranks[i, sorted_indices[2]] = 1  # 最大值

    # 3. 计算平均排名 (axis=0 => 对每列求平均)
    avg_ranks = np.mean(ranks, axis=0)  # 形状 (3,)

    # 4. 统计每个矩阵在 num_tasks 个任务中 rank=1,2,3 的次数
    rank1_counts = np.sum(ranks == 1, axis=0)  # (3,)
    rank2_counts = np.sum(ranks == 2, axis=0)
    rank3_counts = np.sum(ranks == 3, axis=0)

    # 转换为概率(出现次数 / 任务数)
    rank1_probs = rank1_counts / num_tasks
    rank2_probs = rank2_counts / num_tasks
    rank3_probs = rank3_counts / num_tasks

    # 5. 打印结果
    print("cm1 对角线:", diag1)
    print("cm2 对角线:", diag2)
    print("cm3 对角线:", diag3)
    print("\nranks (每行对应第 i 个任务下 [cm1, cm2, cm3] 的排名):\n", ranks)
    print("\n平均排名 (数值越小表示越好):", avg_ranks)
    print("rank=1 次数:", rank1_counts, " 概率:", rank1_probs)
    print("rank=2 次数:", rank2_counts, " 概率:", rank2_probs)
    print("rank=3 次数:", rank3_counts, " 概率:", rank3_probs)

    # 6. 上传到W&B
    wandb.log({
        # 平均排名
        "Average Rank/cm1": float(avg_ranks[0]),
        "Average Rank/cm2": float(avg_ranks[1]),
        "Average Rank/cm3": float(avg_ranks[2]),

        # rank=1 概率
        "Rank1 Prob/cm1": float(rank1_probs[0]),
        "Rank1 Prob/cm2": float(rank1_probs[1]),
        "Rank1 Prob/cm3": float(rank1_probs[2]),

        # rank=2 概率
        "Rank2 Prob/cm1": float(rank2_probs[0]),
        "Rank2 Prob/cm2": float(rank2_probs[1]),
        "Rank2 Prob/cm3": float(rank2_probs[2]),

        # rank=3 概率
        "Rank3 Prob/cm1": float(rank3_probs[0]),
        "Rank3 Prob/cm2": float(rank3_probs[1]),
        "Rank3 Prob/cm3": float(rank3_probs[2]),
    })


    # 你也可以 return avg_ranks, rank1_probs, ... 等结果
    return avg_ranks, rank1_probs, rank2_probs, rank3_probs