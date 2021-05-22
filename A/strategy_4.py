import sys
input = sys.stdin.readline
import math
import heapq
import random
import logging
LEFT = 0
TOP = 1
RIGHT = 2
BOTTOM = 3


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read():
    N = int(input().strip())
    IXYR = []
    for i in range(N):
        x, y, r = map(int, input().strip().split())
        IXYR.append((i, x, y, r))
    return N, IXYR


def rect_size(rect):
    (_, left, top, right, bottom) = rect
    return (right - left) * (bottom - top)


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


def list_divisors(n, sorted = False):
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


def search_adspace(i, x, y, r, ads, T=20, U=5):
    (_, left, top, right, bottom) = ads[i]
    s = rect_size(ads[i])
    ar = r
    for t in range(T):
        ar = s + (r - s) * math.cos(math.pi * t / (T << 1))
        div = list_divisors(int(ar))
        #random.shuffle(div)
        for width in div[::-1]:
            height = int(ar) // width
            for _ in range(U):
                factor_l = random.random()
                factor_t = random.random()
                left = math.floor(x - width * factor_l)
                top = math.floor(y - height * factor_t)
                new_ad = (i, left, top, left + width, top + height)
                if point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                    ads[i] = new_ad
                    return


def get_score(r, s):
    return 1 - (1 - min(r, s) / max(r, s))**2


def get_total_score(N, IXYR, ads):
    return 10**9 * sum([get_score(x[3], rect_size(ads[x[0]])) for x in IXYR]) / N


def erode_threshold(i, x, y, r, ads, factor=0.9, direction=-1, th=0.0, inverse=False):
    s = rect_size(ads[i])
    score = get_score(r, s)
    if inverse ^ (score < th):
        erode(i, x, y, r, ads, factor, direction)


def direction_decomposition(cx, cy, ox, oy):
    # 点(ox, oy) から見た (cx, cy) の向きを返す
    vx = RIGHT if cx > ox else LEFT
    vy = BOTTOM if cy > oy else TOP
    return vx, vy


def translate(i, x, y, r, ad, ads, direction=-1):
    # [単位操作] 平行移動
    j, left, top, right, bottom = ad
    if direction == -1:
        direction = random.randint(0, 3)
    dx, dy = 0, 0
    if direction == LEFT:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (i, left - mid, top, right - mid, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            dx = -ok
    elif direction == TOP:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (i, left, top - mid, right, bottom - mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            dy = -ok
    elif direction == RIGHT:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (i, left + mid, top, right + mid, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            dx = ok
    else:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (i, left, top + mid, right, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            dy = ok
    return (j, left + dx, top + dy, right + dx, bottom + dy)


def dilate(i, x, y, r, ad, ads, direction=-1):
    # [単位操作] 膨張
    j, left, top, right, bottom = ad
    if direction == -1:
        direction = random.randint(0, 3)
    if direction == LEFT:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (i, left - mid, top, right, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            left -= ok
    elif direction == TOP:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (i, left, top - mid, right, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            top -= ok
    elif direction == RIGHT:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (i, left, top, right + mid, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            right += ok
    else:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (i, left, top, right, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            bottom += ok
    return (j, left, top, right, bottom)


def erode(i, x, y, r, ad, ads, factor=0.9, direction=-1):
    # [単位操作] 縮小
    j, left, top, right, bottom = ad
    if direction == -1:
        direction = random.randint(0, 3)
    if direction == LEFT:
        left = max(x, int(left + (x - left) * factor))
    elif direction == TOP:
        height = bottom - top
        top = max(y, int(top + (y - top) * factor))
    elif direction == RIGHT:
        width = right - left
        right = min(x, int(right - (right - x) * factor)) + 1
    else:
        height = bottom - top
        bottom = min(y, int(bottom - (bottom - y) * factor)) + 1
    return (j, left, top, right, bottom)


def reshape(i, x, y, r, ad, ads, direction=-1, corner=-1):
    # [単位操作] 等積変形
    j, left, top, right, bottom = ad
    s = rect_size(ad)
    if direction == -1:
        direction = random.randint(0, 3)
    if corner == -1:
        corner = random.randint(0, 1)
    # 1. left を mid だけ増やして bottom を減らす
    if direction == LEFT and corner == 0:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            width = right - left + mid
            height = int(s / width)
            new_ad = (j, left - mid, top, right, top + height)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            width = right - left + ok
            height = int(s / width)
            new_ad = (j, left - ok, top, right, top + height)
    # 2. left を mid だけ増やして top を減らす
    elif direction == LEFT and corner == 1:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            width = right - left + mid
            height = int(s / width)
            new_ad = (j, left - mid, bottom - height, right, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            width = right - left + ok
            height = int(s / width)
            new_ad = (j, left - ok, bottom - height, right, bottom)
    # 3. top を mid だけ増やして left を減らす
    elif direction == TOP and corner == 0:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            height = bottom - top + mid
            width = int(s / height)
            new_ad = (j, right - width, top - mid, right, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            height = bottom - top + ok
            width = int(s / height)
            new_ad = (j, right - width, top - ok, right, bottom)
    # 4. top を mid だけ増やして right を減らす
    elif direction == TOP and corner == 1:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            height = bottom - top + mid
            width = int(s / height)
            new_ad = (j, left, top - mid, left + width, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            height = bottom - top + ok
            width = int(s / height)
            new_ad = (j, left, top - ok, left + width, bottom)
    # 5. right を mid だけ増やして top を減らす
    elif direction == RIGHT and corner == 0:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            width = right - left + mid
            height = int(s / width)
            new_ad = (j, left, bottom - height, right + mid, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            width = right - left + ok
            height = int(s / width)
            new_ad = (j, left, bottom - height, right + ok, bottom)
    # 6. right を mid だけ増やして bottom を減らす
    elif direction == RIGHT and corner == 1:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            width = right - left + mid
            height = int(s / width)
            new_ad = (j, left, top, right + mid, top + height)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            width = right - left + ok
            height = int(s / width)
            new_ad = (j, left, top, right + ok, top + height)
    # 7. bottom を mid だけ増やして right を減らす
    elif direction == BOTTOM and corner == 0:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            height = bottom - top + mid
            width = int(s / height)
            new_ad = (j, left, top, left + width, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            height = bottom - top + ok
            width = int(s / height)
            new_ad = (j, left, top, left + width, bottom + ok)
    # 8. bottom を mid だけ増やして left を減らす
    else:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            height = bottom - top + mid
            width = int(s / height)
            new_ad = (j, right - width, top, right, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            height = bottom - top + ok
            width = int(s / height)
            new_ad = (j, right - width, top, right, bottom + ok)
    return new_ad


def sortby_score(N, IXYR, ads, index=False):
    if index:
        return [i[0] for i in sorted(enumerate(IXYR), key=lambda x: get_score(x[1][3], rect_size(ads[x[1][0]])))]
    return list(sorted(IXYR, key=lambda x: get_score(x[3], rect_size(ads[x[0]]))))


def sortby_diffusion_v1(N, IXYR, ads, ox, oy, index=False):
    if index:
        return [i[0] for i in sorted(enumerate(IXYR), key=lambda x: (x[1][1] - ox)**2 + (x[1][2] - oy)**2)]
    return list(sorted(IXYR, key=lambda x: (x[1] - ox)**2 + (x[2] - oy)**2))


def sortby_diffusion_v2(N, IXYR, ads, ox, oy, index=False):
    if index:
        return [i[0] for i in sorted(enumerate(IXYR), key=lambda x: ((ads[x[1][0]][1] + ads[x[1][0]][3])//2 - ox)**2 + ((ads[x[1][0]][2] + ads[x[1][0]][4])//2 - oy)**2)]
    return list(sorted(IXYR, key=lambda x: ((ads[x[0]][1] + ads[x[0]][3])//2 - ox)**2 + ((ads[x[0]][2] + ads[x[0]][4])//2 - oy)**2))


def annealing(N, IXYR, ads, n=20000, T0=1e6, T1=1e2):
    for t in range(n):
        T = pow(T0, 1 - t/n) * pow(T1, t/n)
        k = random.randint(0, N-1)
        i, x, y, r = IXYR[k]
        old_score = get_score(r, rect_size(ads[i]))
        new_ad = ads[i]
        a = random.random()
        if a < 0.5:
            new_ad = reshape(i, x, y, r, new_ad, ads)
        else:
            new_ad = translate(i, x, y, r, new_ad, ads)
        new_ad = dilate(i, x, y, r, new_ad, ads)
        new_score = get_score(r, rect_size(new_ad))
        delta = new_score - old_score
        q = pow(math.e, delta / T)
        p = 1 if delta >= 0 else q
        if random.random() < p:
            # logger.info("delta: {:.06f}, p: {:.06f}".format(delta, p))
            ads[i] = new_ad


def solve(N, IXYR):
    # 0. set 1px ads
    ads = []
    for i, x, y, r in IXYR:
        ads.append((i, x, y, x+1, y+1))
    
    # 1. search from largest
    IXYR.sort(key=lambda x: -x[3])
    for i, x, y, r in IXYR:
        search_adspace(i, x, y, r, ads, T=40, U=30)
    logger.info("search_adspace: {:.06f}".format(get_total_score(N, IXYR, ads)))
    
    # 2. annealing
    annealing(N, IXYR, ads)
    logger.info("annealing: {:.06f}".format(get_total_score(N, IXYR, ads)))

    # 3. dilate
    for i, x, y, r in IXYR:
        for d in random.sample(range(4), k=4):
            ad = ads[i]
            ads[i] = dilate(i, x, y, r, ad, ads, direction=d)

    
    logger.info("total: {:.06f}".format(get_total_score(N, IXYR, ads)))
    
    for i, a, b, c, d in sorted(ads):
        print(a, b, c, d)


if __name__ == '__main__':
    # random.seed(2021)
    inputs = read()
    solve(*inputs)
