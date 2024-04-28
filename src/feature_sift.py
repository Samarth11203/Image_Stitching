# coding: utf-8

import cv2
import numpy as np
import constant as const

# Initialize SIFT
sift = cv2.SIFT_create()

def detect_sift_features(img):
    """
    Detect keypoints and descriptors using SIFT.
    Args:
        img: input image
    Returns:
        keypoints, descriptors
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    positions = [[kp.pt[1], kp.pt[0]] for kp in keypoints]
    return descriptors, positions

def matching(descriptor1, descriptor2, feature_position1, feature_position2, pool, y_range=10):
    """
    Matching two groups of SIFT descriptors with a y-range restriction.
    """
    TASKS_NUM = 32

    partition_descriptors = np.array_split(descriptor1, TASKS_NUM)
    partition_positions = np.array_split(feature_position1, TASKS_NUM)

    sub_tasks = [(partition_descriptors[i], descriptor2, partition_positions[i], feature_position2, y_range) for i in range(TASKS_NUM)]
    results = pool.starmap(compute_match, sub_tasks)
    
    matched_pairs = []
    for res in results:
        if len(res) > 0:
            matched_pairs += res

    return matched_pairs

def compute_match(descriptor1, descriptor2, feature_position1, feature_position2, y_range=10):
    matched_pairs = []
    matched_pairs_rank = []
    
    for i in range(len(descriptor1)):
        distances = []
        y = feature_position1[i][0]
        for j in range(len(descriptor2)):
            diff = float('Inf')
            
            # Only compare features with similar y-axis 
            if y-y_range <= feature_position2[j][0] <= y+y_range:
                diff = descriptor1[i] - descriptor2[j]
                diff = (diff**2).sum()
            distances += [diff]

        sorted_index = np.argpartition(distances, 1)
        local_optimal = distances[sorted_index[0]]
        local_optimal2 = distances[sorted_index[1]]
        if local_optimal > local_optimal2:
            local_optimal, local_optimal2 = local_optimal2, local_optimal
        
        if local_optimal/local_optimal2 <= 0.5:
            paired_index = np.where(distances == local_optimal)[0][0]
            pair = [feature_position1[i], feature_position2[paired_index]]
            matched_pairs += [pair]
            matched_pairs_rank += [local_optimal]

    # Refine pairs
    sorted_rank_idx = np.argsort(matched_pairs_rank)
    sorted_match_pairs = np.asarray(matched_pairs)
    sorted_match_pairs = sorted_match_pairs[sorted_rank_idx]

    refined_matched_pairs = []
    for item in sorted_match_pairs:
        duplicated = False
        for refined_item in refined_matched_pairs:
            if refined_item[1] == list(item[1]):
                duplicated = True
                break
        if not duplicated:
            refined_matched_pairs += [item.tolist()]
            
    return refined_matched_pairs
