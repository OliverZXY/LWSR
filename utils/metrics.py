import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, kendalltau


def rank_retrievals(query_vectors, retrieval_library):
    """根据距离矩阵对检索进行排名"""
    dist_matrix = cdist(query_vectors, retrieval_library, 'euclidean')
    ranks = np.argsort(dist_matrix, axis=1)
    return ranks


def mean_absolute_rank_change(rank1, rank2):
    """计算平均绝对排名变化"""
    total_absolute_rank_changes = 0

    for i in range(rank1.shape[0]):
        uuid_array_1 = rank1[i]
        uuid_array_2 = rank2[i]

        rank_dict_1 = {uuid: rank for rank, uuid in enumerate(uuid_array_1)}
        rank_dict_2 = {uuid: rank for rank, uuid in enumerate(uuid_array_2)}

        absolute_rank_changes = np.abs(np.array([rank_dict_1[uuid] - rank_dict_2[uuid] for uuid in uuid_array_1]))
        average_absolute_rank_change = np.mean(absolute_rank_changes)

        total_absolute_rank_changes += average_absolute_rank_change

    result = total_absolute_rank_changes / rank1.shape[0]

    return result


def spearman_rank_correlation(rank1, rank2):
    """计算斯皮尔曼等级相关系数"""
    spearman_coefficients = []

    for i in range(rank1.shape[0]):
        uuid_array_1 = rank1[i]
        uuid_array_2 = rank2[i]

        rank_dict_1 = {uuid: rank for rank, uuid in enumerate(uuid_array_1)}
        rank_dict_2 = {uuid: rank for rank, uuid in enumerate(uuid_array_2)}

        ranks_1 = np.array([rank_dict_1[uuid] for uuid in uuid_array_1])
        ranks_2 = np.array([rank_dict_2[uuid] for uuid in uuid_array_1])
        spearman_corr, _ = spearmanr(ranks_1, ranks_2)
        spearman_coefficients.append(spearman_corr)

    average_spearman_coefficient = np.mean(spearman_coefficients)

    return average_spearman_coefficient


def kendall_rank_correlation(rank1, rank2):
    """计算肯德尔等级相关系数"""
    kendall_coefficients = []

    for i in range(rank1.shape[0]):
        uuid_array_1 = rank1[i]
        uuid_array_2 = rank2[i]

        rank_dict_1 = {uuid: rank for rank, uuid in enumerate(uuid_array_1)}
        rank_dict_2 = {uuid: rank for rank, uuid in enumerate(uuid_array_2)}

        ranks_1 = np.array([rank_dict_1[uuid] for uuid in uuid_array_1])
        ranks_2 = np.array([rank_dict_2[uuid] for uuid in uuid_array_1])

        kendall_corr, _ = kendalltau(ranks_1, ranks_2)
        kendall_coefficients.append(kendall_corr)

    average_kendall_coefficient = np.mean(kendall_coefficients)

    return average_kendall_coefficient


def select_uuids(uuids, ranks):
    uuid_array = np.array(uuids)
    selected_uuids = uuid_array[ranks]

    return selected_uuids


def calc_all_rank_metrics(query_vectors, retrieval_library, uuids, initial_uuids):
    updated_ranks = rank_retrievals(query_vectors, retrieval_library)
    updated_uuids = select_uuids(uuids, updated_ranks)
    marc = mean_absolute_rank_change(initial_uuids, updated_uuids)
    src = spearman_rank_correlation(initial_uuids, updated_uuids)
    krc = kendall_rank_correlation(initial_uuids, updated_uuids)

    return marc, src, krc, updated_uuids


def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)
