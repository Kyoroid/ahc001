from __future__ import annotations
import sys
input = sys.stdin.readline
import math
import random
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read():
    N = int(input().strip())
    IXYR = []
    for i in range(N):
        x, y, r = map(int, input().strip().split())
        IXYR.append((i, x, y, r))
    return N, IXYR


def point_in_rect(x, y, rect):
    (_, left, top, right, bottom) = rect
    return left <= x < right and top <= y < bottom


def rect_in_field(rect):
    (_, left, top, right, bottom) = rect
    return 0 <= left and right <= 10000 and 0 <= top and bottom <= 10000


def rect_in_rect(rect0, rect1):
    (i, left0, top0, right0, bottom0) = rect0
    (j, left1, top1, right1, bottom1) = rect1
    if i == j:
        return False
    cx0 = right0 + left0
    cx1 = right1 + left1
    cy0 = bottom0 + top0
    cy1 = bottom1 + top1
    diff_cx = abs(cx1 - cx0)
    diff_cy = abs(cy1 - cy0)
    th_x = (right0 - left0) + (right1 - left1)
    th_y = (bottom0 - top0) + (bottom1 - top1)
    return diff_cx <= th_x and diff_cy <= th_y


def check_adspace(ad, placed_ads):
    if not rect_in_field(ad):
        return False
    for placed_ad in placed_ads:
        if rect_in_rect(ad, placed_ad):
            return False
    return True


def list_divisors(n: int, sorted: bool = False) -> list[int]:
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i * i != n:
                divisors.append(n // i)
        i += 1
    if sorted:
        divisors.sort()
    return divisors


def search_adspace(i, x, y, r, ads):
    (_, left, top, right, bottom) = ads[i]
    s = (bottom - top) * (right - left)
    # 1. reduce
    ar = r
    while ar > s:
        # 1.1. divisor
        div = list_divisors(int(ar))
        random.shuffle(div)
        for width in div:
            height = int(ar) // width
            for _ in range(5):
                factor_l = random.random()
                factor_t = random.random()
                left = math.floor(x - width * factor_l)
                top = math.floor(y - height * factor_t)
                ad = (i, left, top, left + width, top + height)
                if point_in_rect(x, y, ad) and check_adspace(ad, ads):
                    return ad
        ar = max(1, ar * 0.85)
    return ads[i]


def dilate_bisect(i, x, y, r, ads):
    ad = ads[i]

    # dilate right
    _, left, top, right, bottom = ad
    ok, ng = 0, 10000 - (right - left)
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        new_ad = (i, left, top, right + mid, bottom)
        s = (right + mid - left) * (bottom - top)
        if s <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
            ok = mid
        else:
            ng = mid
    new_ad = (i, left, top, right + ok, bottom)
    s = (right + ok - left) * (bottom - top)
    if s <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
        ad = new_ad

    # dilate bottom
    _, left, top, right, bottom = ad
    ok, ng = 0, 10000 - (right - left)
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        new_ad = (i, left, top, right, bottom + mid)
        s = (right - left) * (bottom + mid - top)
        if s <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
            ok = mid
        else:
            ng = mid
    new_ad = (i, left, top, right, bottom + ok)
    s = (right - left) * (bottom + ok - top)
    if s <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
        ad = new_ad

    # dilate left
    _, left, top, right, bottom = ad
    ok, ng = 0, 10000 - (right - left)
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        new_ad = (i, left - mid, top, right, bottom)
        s = (right - left + mid) * (bottom - top)
        if s <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
            ok = mid
        else:
            ng = mid
    new_ad = (i, left - ok, top, right, bottom)
    s = (right - left - ok) * (bottom - top)
    if s <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
        ad = new_ad

    # dilate top
    _, left, top, right, bottom = ad
    ok, ng = 0, 10000 - (right - left)
    while abs(ok - ng) > 1:
        mid = (ok + ng) // 2
        new_ad = (i, left, top - mid, right, bottom)
        s = (right - left) * (bottom - top + mid)
        if s <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
            ok = mid
        else:
            ng = mid
    new_ad = (i, left, top - ok, right, bottom)
    s = (right - left) * (bottom - top + ok)
    if s <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
        ad = new_ad
    return ad


def solve(N, IXYR):
    # 0. set 1px ads
    ads = []
    for i, x, y, r in IXYR:
        ads.append((i, x, y, x+1, y+1))
    # iterate large ad first
    IXYR.sort(key=lambda x: -x[3])
    for i, x, y, r in IXYR:
        ads[i] = search_adspace(i, x, y, r, ads)
    
    IXYR.sort(key=lambda x: x[1] + x[2])
    for i, x, y, r in IXYR:
        ads[i] = dilate_bisect(i, x, y, r, ads)
    
    IXYR.sort(key=lambda x: x[1] + x[2])
    for i, x, y, r in IXYR:
        ads[i] = dilate_bisect(i, x, y, r, ads)
        
    for i, a, b, c, d in sorted(ads):
        print(a, b, c, d)


if __name__ == '__main__':
    random.seed(4)
    inputs = read()
    solve(*inputs)
