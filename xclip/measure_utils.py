import numpy as np

def mrr_score(similarities, ground_truth_indices, k_list = [1, 3, 5, 10]):
    """
    Calculate the Mean Reciprocal Rank (MRR) at multiple values of k for a set of embeddings.

    Parameters:
    - similarities (np.array): A 2D numpy array where each row represents the similarity scores
      between a video embedding and all text embeddings.
    - ground_truth_indices (np.array): An array where each element is the index of the ground truth
      text embedding for the corresponding video embedding.
    - k_list (list): A list of integers specifying the k values to calculate MRR for.

    Returns:
    - dict: A dictionary where keys are k values and values are the MRR at each k.
    """
    sorted_indices = np.argsort(-similarities, axis=1)
    mrr_scores = {}

    for k in k_list:
        reciprocal_ranks = []

        for i, sorted_index_list in enumerate(sorted_indices):
            ranks = np.where(sorted_index_list[:k] == ground_truth_indices[i])[0]
            if ranks.size > 0:
                # If found within the top k, append the reciprocal rank
                reciprocal_ranks.append(1 / (ranks[0] + 1))
            else:
                # If not found within the top k, append 0
                reciprocal_ranks.append(0)

        # Calculate mean of the reciprocal ranks
        mrr_scores[f"Top {k}"] = np.mean(reciprocal_ranks) #求平均

    return mrr_scores

def check_pairs(
    reduced_video_embeddings: np.array,
    reduced_text_embeddings: np.array,
    mappings,
    small_scale=True,
):
    """
    Checks the pairing accuracy between video and text embeddings by finding
    the nearest text embedding for each video embedding and see whether it is the corresponding text of the video.

    Prints the accuracy of correct pairings and the indices of video embeddings that are incorrectly paired.
    """
    distances = np.linalg.norm(
        reduced_video_embeddings[:, np.newaxis, :]
        - reduced_text_embeddings[np.newaxis, :, :],
        axis=2,
    )
    #similarities = reduced_video_embeddings @ reduced_text_embeddings.T
    similarities = reduced_text_embeddings @ reduced_video_embeddings.T
    #similarities = (similarity_score(reduced_video_embeddings, reduced_text_embeddings)).numpy()
    sorted_text_indices = np.argsort(-similarities, axis=1)


    accuracies = {}
    mrr = {}
    if small_scale:
        video_id_to_text_label = mappings["video_id_to_text_label"]
        index_to_video_id = mappings["index_to_video_id"]
        index_to_text_label = mappings["index_to_text_label"]
        ground_truth_indices = np.arange(len(reduced_video_embeddings))
        mrr_k = mrr_score(similarities, ground_truth_indices, k_list=[1, 3, 5])
        for n in [1, 3, 5]:
            correct_pairs = np.array(
                [
                    ground_truth in sorted_text_indices[i, :n]
                    for i, ground_truth in enumerate(ground_truth_indices)
                ]
            )
            accuracy = np.mean(correct_pairs)
            accuracies[f"Top {n}"] = round(accuracy * 100, 4)
            print(f"Top {n} accuracy: {accuracy * 100:.2f}%")
        top_1_indices = sorted_text_indices[:, 0]
        correct_top_1_pairs = top_1_indices == ground_truth_indices
        incorrect_top_1_indices = np.where(~correct_top_1_pairs)[0]

        incorrect_pair_video_id = [
            mappings["index_to_video_id"][i] for i in incorrect_top_1_indices
        ]
        print(
            f"IDs of incorrectly paired video embeddings (Top 1): {incorrect_pair_video_id}"
        )

        for i, indices in enumerate(sorted_text_indices):
            video_id = index_to_video_id[i]
            original_text_label = video_id_to_text_label[video_id]
            sorted_text_labels = [index_to_text_label[j] for j in indices]
            print(f"Video {video_id}:")
            print(f"Ground truth text label: {original_text_label}")
            print(f"Sorted Matching text labels: {sorted_text_labels}")
    else:
        ground_truth_indices = np.arange(len(reduced_video_embeddings))
        mrr_k = mrr_score(similarities, ground_truth_indices, k_list = [1,3,5,10])
        cumulative_accuracy = np.zeros(len(reduced_video_embeddings))
        top_n_values = [1, 3, 5, 10]
        for n in top_n_values:
            for i in range(len(sorted_text_indices)):
                if ground_truth_indices[i] in sorted_text_indices[i, :n]:
                    cumulative_accuracy[i] = 1
            accuracy_n = np.mean(cumulative_accuracy)
            accuracies[f"Top {n}"] = round(accuracy_n * 100, 4)
            # print(f"Accuracy within top {n}: {accuracy_n * 100:.2f}%")
    return accuracies, mrr_k
