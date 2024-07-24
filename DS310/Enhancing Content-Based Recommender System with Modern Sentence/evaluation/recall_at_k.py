# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import torch
# import concurrent.futures
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split


# def extract_best_indices(m, topk, mask=None):
#     """
#     Use sum of the cosine distance over all tokens ans return best mathes.
#     m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
#     topk (int): number of indices to return (from high to lowest in order)
#     """
#     # return the sum on all tokens of cosinus for each sentence
#     if len(m.shape) > 1:
#         cos_sim = np.mean(m, axis=0)
#     else:
#         cos_sim = m
#     index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score
#     if mask is not None:
#         assert mask.shape == m.shape
#         mask = mask[index]
#     else:
#         mask = np.ones(len(cos_sim))
#     mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
#     best_index = index[mask][:topk]
#     return best_index


# def get_recall_at_k_worker(user, test_data, cluster, embed_mat, k_values):
#     user_data = test_data[test_data.user_id == user]
#     ## split to train and test
#     train_user_data, test_user_data = train_test_split(user_data, test_size=0.2, random_state=42)
#     train_relevant_items = train_user_data[train_user_data.rating > 3]['index'].tolist()
#     test_relevant_items = test_user_data[test_user_data.rating > 3]['index'].tolist()

#     if len(train_relevant_items) == 0 or len(test_relevant_items) == 0:
#         return []

#     # train_relevant_items_topics = np.array([cluster[i] for i in train_relevant_items])
#     predict_vec = torch.mean(embed_mat[train_relevant_items], dim=0).reshape(1, -1)
#     mat = cosine_similarity(predict_vec, embed_mat)

#     predict_indices = extract_best_indices(mat, topk=max(k_values))
#     predict_items_topics = np.array([cluster[i] for i in predict_indices])
#     test_relevant_items_topic = np.array([cluster[i] for i in test_relevant_items])

#     recall_user = []
#     for k in k_values:
#         intersection = len(set(predict_items_topics[:k]) & set(test_relevant_items_topic))
#         recall_at_k = intersection / len(test_relevant_items_topic)
#         recall_user.append(recall_at_k)

#     return np.array(recall_user)

# def get_recall_at_k_parallel(test_data, cluster, embed_mat):
#     k_values = [5, 10, 50]
#     recall_list = []
#     user_ids = test_data.user_id.unique()
#     random_user_ids = np.random.choice(user_ids, size=2000, replace=False)

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(get_recall_at_k_worker, user, test_data, cluster, embed_mat, k_values) for user in random_user_ids]

#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#             result = future.result()
#             if len(result) > 0:
#                 recall_list.append(result)

#     return np.array(recall_list)


# def recall_at_k(test_data, cluster, embed_mat):
#     recall_list = get_recall_at_k_parallel(test_data, cluster, embed_mat)
#     recall_at_k = np.mean(recall_list, axis=0)
#     return recall_at_k


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import concurrent.futures
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens ans return best mathes.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]
    return best_index


def get_recall_at_k_worker(user, test_data, cluster, embed_mat, k_values):
    user_data = test_data[test_data.user_id == user]
    ## split to train and test
    try:
        train_user_data, test_user_data = train_test_split(user_data, test_size=0.2, train_size=0.8, random_state=42)
    except:
        return []
    train_relevant_items = train_user_data[train_user_data.rating > 3]['index'].tolist()
    test_relevant_items = test_user_data[test_user_data.rating > 3]['index'].tolist()

    if len(train_relevant_items) == 0 or len(test_relevant_items) == 0:
        return []

    # train_relevant_items_topics = np.array([cluster[i] for i in train_relevant_items])
    predict_vec = torch.mean(embed_mat[train_relevant_items], dim=0).reshape(1, -1)
    mat = cosine_similarity(predict_vec, embed_mat)

    predict_indices = extract_best_indices(mat, topk=max(k_values))
    predict_items_topics = np.array([cluster[i] for i in predict_indices])
    test_relevant_items_topic = np.array([cluster[i] for i in test_relevant_items])

    recall_user = []
    for k in k_values:
        intersection = len(set(predict_items_topics[:k]) & set(test_relevant_items_topic))
        recall_at_k = intersection / len(test_relevant_items_topic)
        recall_user.append(recall_at_k)

    return np.array(recall_user)

def get_recall_at_k_parallel(test_data, cluster, embed_mat, size, k_list):
    k_values = k_list
    recall_list = []
    user_ids = test_data.user_id.unique()
    random_user_ids = np.random.choice(user_ids, size=size, replace=False)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_recall_at_k_worker, user, test_data, cluster, embed_mat, k_values) for user in random_user_ids]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if len(result) > 0:
                recall_list.append(result)

    return np.array(recall_list)


def recall_at_k(test_data, cluster, embed_mat):
    recall_list = get_recall_at_k_parallel(test_data, cluster, embed_mat)
    recall_at_k = np.mean(recall_list, axis=0)
    return recall_at_k