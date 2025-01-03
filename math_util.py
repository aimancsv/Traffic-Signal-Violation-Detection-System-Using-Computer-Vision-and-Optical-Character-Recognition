from dataclasses import dataclass
import numpy as np

@dataclass
class BoundBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    classes: list = None
    label: int = -1
    score: float = -1

    def __post_init__(self):
        if self.classes is not None:
            self.label = np.argmax(self.classes)  # Find the index of the maximum value in classes
            self.score = self.classes[self.label]  # Assign the maximum value as the score

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0  # No overlap between the intervals
        else:
            return min(x2, x4) - x1  # Calculate the overlap length
    else:
        if x2 < x3:
            return 0  # No overlap between the intervals
        else:
            return min(x2, x4) - x3  # Calculate the overlap length

def intersection(p, q, r, t):
    # Find the intersection point of two lines (p-q and r-t)

    (x1, y1) = p
    (x2, y2) = q

    (x3, y3) = r
    (x4, y4) = t

    # Calculate the coefficients of the line equations
    a1 = y1 - y2
    b1 = x2 - x1
    c1 = x1 * y2 - x2 * y1

    a2 = y3 - y4
    b2 = x4 - x3
    c2 = x3 * y4 - x4 * y3

    if a1 * b2 - a2 * b1 == 0:
        return False  # Lines are parallel, no intersection

    # Calculate the intersection point coordinates
    x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
    y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)

    if x1 > x2:
        # Swap x1 and x2 if x1 is greater than x2
        tmp = x1
        x1 = x2
        x2 = tmp
    if y1 > y2:
        # Swap y1 and y2 if y1 is greater than y2
        tmp = y1
        y1 = y2
        y2 = tmp
    if x3 > x4:
        # Swap x3 and x4 if x3 is greater than x4
        tmp = x3
        x3 = x4
        x4 = tmp
    if y3 > y4:
        # Swap y3 and y4 if y3 is greater than y4
        tmp = y3
        y3 = y4
        y4 = tmp

    # Check if the intersection point lies within both line segments
    return x1 <= x <= x2 and y1 <= y <= y2 and x3 <= x <= x4 and y3 <= y <= y4
