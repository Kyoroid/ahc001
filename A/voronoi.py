import sys
input = sys.stdin.readline
import math
import random
import logging
LEFT = 0
TOP = 1
RIGHT = 2
BOTTOM = 3


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def read():
    N = int(input().strip())
    XYR = []
    for i in range(N):
        XYR.append(list(map(int, input().strip().split())))
    return N, XYR


def solve(N, XYR):
    M = 100000
    ZWL = []
    # seeding
    for i in range(M):
        rx = random.uniform(0, 10000)
        ry = random.uniform(0, 10000)
        ZWL.append([rx, ry, -1])
    # make cluster
    for i in range(M):
        rx, ry, _ = ZWL[i]
        label = -1
        min_dist = 10**9+1
        for j in range(N):
            x, y, r = XYR[i]
            dist = abs(ry - y) + abs(rx - x)
            if min_dist > dist:
                min_dist = dist
                label = j
        ZWL[i][2] = label
    # dilate
    for j in range(N):
        x, y, r = XYR[i]
        left, top, right, bottom = x, y, x+1, y+1




if __name__ == '__main__':
    # random.seed(202)
    inputs = read()
    solve(*inputs)