# coding: utf-8

import cv2
import numpy as np
import constant as const

def detect_shi_tomasi_features(img, max_corners=5000, quality_level=0.01, min_distance=10):
    """
    Detect keypoints using Shi-Tomasi corner detection.
    Args:
        img: input image
        max_corners: maximum number of corners to detect
        quality_level: minimal accepted quality of corners (relative to the best corner)
        min_distance: minimum distance between detected corners
    Returns:
        keypoints, positions
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance
    )
    
    if corners is not None:
        corners = np.int0(corners)  # Convert to integer
        positions = [[c[0][1], c[0][0]] for c in corners]
        descriptors = np.array([img[y, x].flatten() for (y, x) in positions])
        return descriptors, positions
    else:
        return None, None

def matching(descriptor1, descriptor2, feature_position1, feature_position2, pool, y_range=10):
    """
    Matching two groups of descriptors with a y-range restriction.
    """
    TASKS_NUM = 32  # Can adjust based on available processing cores

    # Partition descriptors and positions for parallel processing
    partition_descriptors = np.array_split(descriptor1, TASKS_NUM)
    partition_positions = np.array_split(feature_position1, TASKS_NUM)

    sub_tasks = [
        (partition_descriptors[i], descriptor2, partition_positions[i], feature_position2, y_range)
        for i in range(TASKS_NUM)
    ]

    results = pool.starmap(compute_match, sub_tasks)

    matched_pairs = []
    for res in results:
        if len(res) > 0:
            matched_pairs += res

    return matched_pairs

def compute_match(descriptor1, descriptor2, feature_position1, feature_position2, y_range=10):
    """
    Compute matching pairs with a y-range constraint.
    """
    matched_pairs = []
    matched_pairs_rank = []

    for i in range(len(descriptor1)):
        distances = []
        y = feature_position1[i][0]
        for j in range(len(descriptor2)):
            diff = float('Inf')

            # Only compare features with a similar y-axis alignment
            if y - y_range <= feature_position2[j][0] <= y + y_range:
                diff = np.linalg.norm(descriptor1[i] - descriptor2[j])
            distances.append(diff)

        # Find the best and second-best matches
        sorted_index = np.argpartition(distances, 1)
        local_optimal = distances[sorted_index[0]]
        local_optimal2 = distances[sorted_index[1]]

        if local_optimal > local_optimal2:
            local_optimal, local_optimal2 = local_optimal2, local_optimal

        # Lowe's ratio test
        # Check if local_optimal2 is zero or NaN to avoid invalid calculations
        if local_optimal2 != 0 and not np.isnan(local_optimal2):
            ratio = local_optimal / local_optimal2
            # print(f"ratio: {ratio}")
            if ratio <= 0.5:
                paired_index = sorted_index[0]
                pair = [feature_position1[i], feature_position2[paired_index]]
                matched_pairs.append(pair)
                matched_pairs_rank.append(local_optimal)


    # Refine matched pairs to avoid duplicates
    sorted_rank_idx = np.argsort(matched_pairs_rank)
    sorted_match_pairs = np.asarray(matched_pairs)
    # sorted_match_pairs are sorted according to rank

    refined_matched_pairs = []
    for item in sorted_match_pairs:
        duplicated = False
        for refined_item in refined_matched_pairs:
            if refined_item[1] == list(item[1]):
                duplicated = True
                break
        if not duplicated:
            refined_matched_pairs.append(item.tolist())

    return refined_matched_pairs
