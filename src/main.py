import sys
import cv2
import math
import numpy as np
import multiprocessing as mp
import time  # To calculate elapsed time
import matplotlib.pyplot as plt  # For plotting bar charts

import feature_harris
import feature_sift
import feature_brisk
import feature_orb
import feature_fast
import feature_shiTomasi
import feature_brief
import utils
import stitch
import constant as const


def stitch_images(input_dirname, feature_algo):
    pool = mp.Pool(mp.cpu_count())

    img_list, focal_length = utils.parse(input_dirname)
    cylinder_img_list = img_list.copy()

    stitched_image = cylinder_img_list[0].copy()
    shifts = [[0, 0]]
    cache_feature = [[], []]

    # Initialize variables for storing matched features and computation time
    total_matched_features = 0
    start_time = time.time()

    # Perform stitching for the current algorithm
    for i in range(1, len(cylinder_img_list)):
        img1 = cylinder_img_list[i - 1]
        img2 = cylinder_img_list[i]

        # Get the cached features or extract new ones
        descriptors1, position1 = cache_feature

        if len(descriptors1) == 0:
            if feature_algo == '0':
                corner_response1 = feature_harris.harris_corner(img1, pool)
                descriptors1, position1 = feature_harris.extract_description(
                    img1, corner_response1, kernel=const.DESCRIPTOR_SIZE, threshold=const.FEATURE_THRESHOLD)
            elif feature_algo == '1':
                descriptors1, position1 = feature_sift.detect_sift_features(img1)
            elif feature_algo == '2':
                descriptors1, position1 = feature_brisk.detect_brisk_features(img1)
            elif feature_algo == '3':
                descriptors1, position1 = feature_orb.detect_orb_features(img1)
            elif feature_algo == '4':
                descriptors1, position1 = feature_brief.detect_brief_features(img1)
            elif feature_algo == '5':
                descriptors1, position1 = feature_shiTomasi.detect_shi_tomasi_features(img1)
            elif feature_algo == '6':
                descriptors1, position1 = feature_fast.detect_fast_features(img1)

        # Extract features for img2
        if feature_algo == '0':
            corner_response2 = feature_harris.harris_corner(img2, pool)
            descriptors2, position2 = feature_harris.extract_description(
                img2, corner_response2, kernel=const.DESCRIPTOR_SIZE, threshold=const.FEATURE_THRESHOLD)
        elif feature_algo == '1':
            descriptors2, position2 = feature_sift.detect_sift_features(img2)
        elif feature_algo == '2':
            descriptors2, position2 = feature_brisk.detect_brisk_features(img2)
        elif feature_algo == '3':
            descriptors2, position2 = feature_orb.detect_orb_features(img2)
        elif feature_algo == '4':
            descriptors2, position2 = feature_brief.detect_brief_features(img2)
        elif feature_algo == '5':
            descriptors2, position2 = feature_shiTomasi.detect_shi_tomasi_features(img2)
        elif feature_algo == '6':
            descriptors2, position2 = feature_fast.detect_fast_features(img2)

        # Cache the second image's features for future use
        cache_feature = [descriptors2, position2]

        # Feature matching based on algorithm
        if feature_algo == '0':
            matched_pairs = feature_harris.matching(
                descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE)
        elif feature_algo == '1':
            matched_pairs = feature_sift.matching(
                descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE)
        elif feature_algo == '2':
            matched_pairs = feature_brisk.matching(
                descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE)
        elif feature_algo == '3':
            matched_pairs = feature_orb.matching(
                descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE)
        elif feature_algo == '4':
            matched_pairs = feature_brief.matching(
                descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE)
        elif feature_algo == '5':
            matched_pairs = feature_shiTomasi.matching(
                descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE)
        elif feature_algo == '6':
            matched_pairs = feature_fast.matching(
                descriptors1, descriptors2, position1, position2, pool, y_range=const.MATCHING_Y_RANGE)

        # Accumulate total matched features
        total_matched_features += len(matched_pairs)

        # Find best shift using RANSAC
        shift = stitch.RANSAC(matched_pairs, shifts[-1])
        if len(shift) >= 2:
            shift = [int(round(shift[0])), int(round(shift[1]))]
        else:
            shift = [0, 0]  # Default for invalid shift
        shifts.append(shift)

        # Stitch the image
        stitched_image = stitch.stitching(stitched_image, img2, shift, pool, blending=True, blending_algo='alpha')
        # cv2.imwrite(f'output_{feature_algo}_{i}.jpg', stitched_image)

    # Perform end-to-end alignment
    aligned = stitch.end2end_align(stitched_image, shifts)
    # cv2.imwrite(f'aligned_{feature_algo}.jpg', aligned)

    # Crop the aligned image
    cropped = stitch.crop(aligned)
    cv2.imwrite(f'cropped_{feature_algo}.jpg', cropped)

    # Calculate elapsed time for the stitching process
    total_time = time.time() - start_time

    # Calculate the average number of matched features
    avg_matched_features = total_matched_features / (len(cylinder_img_list) - 1)

    return avg_matched_features, total_time


def main():
    if len(sys.argv) != 2:
        print('[Usage] python script <input img dir>')
        print('[Example] python script ../input_image/parrington')
        sys.exit(0)

    input_dirname = sys.argv[1]

    feature_algorithms = ['0', '1', '2', '3', '4', '5', '6']
    algorithm_names = ["Harris", "SIFT", "BRISK", "ORB", "BRIEF", "Shi-Tomasi", "Fast"]

    avg_matched_features = []
    total_times = []

    # Loop through all feature algorithms
    for i, feature_algo in enumerate(feature_algorithms):
        avg_features, total_time = stitch_images(input_dirname, feature_algo)
        avg_matched_features.append(avg_features)
        total_times.append(total_time)

        print(f'{algorithm_names[i]} - Avg Matched Features: {avg_features:.2f}, Total Time: {total_time:.2f} sec')

    # Create bar plot for average matched features
    plt.figure(figsize=(10, 6))
    plt.bar(algorithm_names, avg_matched_features, color='b')
    plt.xlabel('Feature Algorithms')
    plt.ylabel('Average Matched Features')
    plt.title('Average Matched Features for Different Algorithms')
    plt.savefig('avg_matched_features.png')  # Save plot
    plt.close()  # Close the plot to free memory

    # Create bar plot for total computation time
    plt.figure(figsize=(10, 6))
    plt.bar(algorithm_names, total_times, color='r')
    plt.xlabel('Feature Algorithms')
    plt.ylabel('Total Computation Time (seconds)')
    plt.title('Total Computation Time for Different Algorithms')
    plt.savefig('total_computation_time.png')  # Save plot
    plt.close()  # Close the plot


if __name__ == '__main__':
    main()
