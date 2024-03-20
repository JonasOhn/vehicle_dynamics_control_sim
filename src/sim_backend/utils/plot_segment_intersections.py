"""
 * Simple Vehicle Dynamics Simulator Project
 *
 * Copyright (c) 2023-2024 Authors:
 *   - Jonas Ohnemus <johnemus@ethz.ch>
 *
 * All rights reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
"""

import numpy as np
import matplotlib.pyplot as plt


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # collinear
    elif val > 0:
        return 1  # clockwise
    else:
        return 2  # counterclockwise


def on_segment(p, q, r):
    if (
        q[0] <= max(p[0], r[0])
        and q[0] >= min(p[0], r[0])
        and q[1] <= max(p[1], r[1])
        and q[1] >= min(p[1], r[1])
    ):
        return True
    return False


def intersect(segment1, segment2):
    p1, q1 = segment1
    p2, q2 = segment2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def plot_segments(segments, intersections):
    for segment in segments:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], "b-")

    for intersection in intersections:
        plt.plot(intersection[0], intersection[1], "ro")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Intersection of Random Line Segments")
    plt.grid(True)
    plt.show()


def generate_segments(num_segments, max_coord):
    segments = []
    for _ in range(num_segments):
        p = np.random.randint(0, max_coord, size=(2,))
        q = np.random.randint(0, max_coord, size=(2,))
        segments.append((p, q))
    return segments


def calculate_intersections(segments):
    intersections = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            segment1 = segments[i]
            segment2 = segments[j]
            if intersect(segment1, segment2):
                x1, y1 = segment1[0]
                x2, y2 = segment1[1]
                x3, y3 = segment2[0]
                x4, y4 = segment2[1]
                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denominator != 0:
                    px = (
                        (x1 * y2 - y1 * x2) * (x3 - x4)
                        - (x1 - x2) * (x3 * y4 - y3 * x4)
                    ) / denominator
                    py = (
                        (x1 * y2 - y1 * x2) * (y3 - y4)
                        - (y1 - y2) * (x3 * y4 - y3 * x4)
                    ) / denominator
                    intersections.append((px, py))
    return intersections


def main(num_segments, max_coord):
    segments = generate_segments(num_segments, max_coord)
    intersections = calculate_intersections(segments)
    plot_segments(segments, intersections)


if __name__ == "__main__":
    num_segments = 5
    max_coord = 10
    main(num_segments, max_coord)
