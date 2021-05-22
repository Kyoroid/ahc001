# score: 45876189284
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
        random.shuffle(div)
        for width in div:
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


def translate(i, x, y, r, ads, direction=-1):
    j, left, top, right, bottom = ads[i]
    if direction == -1:
        direction = random.randint(0, 3)
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
            left -= ok
            right -= ok
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
            top -= ok
            bottom -= ok
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
            left += ok
            right += ok
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
            top += ok
            bottom += ok
    ads[i] = (j, left, top, right, bottom)


def dilate(i, x, y, r, ads, direction=-1):
    j, left, top, right, bottom = ads[i]
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
    ads[i] = (j, left, top, right, bottom)


def erode(i, x, y, r, ads, factor=0.9, direction=-1):
    j, left, top, right, bottom = ads[i]
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
    ads[i] = (j, left, top, right, bottom)


def erode_threshold(i, x, y, r, ads, factor=0.9, direction=-1, th=0.0, inverse=False):
    s = rect_size(ads[i])
    score = get_score(r, s)
    if inverse ^ (score < th):
        erode(i, x, y, r, ads, factor, direction)


def direction_decomposition(cx, cy, ox, oy):
    vx = RIGHT if cx > ox else LEFT
    vy = BOTTOM if cy > oy else TOP
    return vx, vy


def reshape(i, x, y, r, ads, U=20):
    div = list_divisors(r)
    random.shuffle(div)
    for width in div:
        height = r // width
        for _ in range(U):
            factor_l = random.random()
            factor_t = random.random()
            left = math.floor(x - width * factor_l)
            top = math.floor(y - height * factor_t)
            new_ad = (i, left, top, left + width, top + height)
            if point_in_rect(x, y, new_ad) and check_adspace(new_ad, ads):
                ads[i] = new_ad
                return True
    return False


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


def solve(N, IXYR):
    # 0. set 1px ads
    ads = []
    for i, x, y, r in IXYR:
        ads.append((i, x, y, x+1, y+1))
    
    # 1. search from largest
    IXYR.sort(key=lambda x: -x[3])
    for i, x, y, r in IXYR:
        search_adspace(i, x, y, r, ads, T=40, U=30)
    
    # 2. dilate
    K = sortby_score(N, IXYR, ads, index=True)
    for k in K:
        i, x, y, r = IXYR[k]
        for d in random.sample(range(4), k=4):
            dilate(i, x, y, r, ads, direction=d)
    
    # 3. diffusion, dilate from lowest score
    K = sortby_score(N, IXYR, ads, index=True)
    for t in range(5):
        for k in K:
            i, ox, oy, r = IXYR[k]
            score = get_score(r, rect_size(ads[i]))

            if score > 0.95:
                break
            
            L = sortby_diffusion_v1(N, IXYR, ads, ox, oy, index=True)
            #L = sortby_diffusion_v2(N, IXYR, ads, ox, oy, index=True)
            for l in L[::-1]:
                j, cx, cy, s = IXYR[l]
                vx, vy = direction_decomposition(cx, cy, ox, oy)
                translate(j, cx, cy, s, ads, direction=vx)
                translate(j, cx, cy, s, ads, direction=vy)
                    
            for d in random.sample(range(4), k=4):
                dilate(i, ox, oy, r, ads, direction=d)
            new_score = get_score(r, rect_size(ads[i]))
        
        K = sortby_score(N, IXYR, ads, index=True)
    
    for i, x, y, r in IXYR:
        for d in random.sample(range(4), k=4):
            dilate(i, x, y, r, ads, direction=d)
    
    
    logger.info("total: {:.06f}".format(get_total_score(N, IXYR, ads)))
    
    for i, a, b, c, d in sorted(ads):
        print(a, b, c, d)


if __name__ == '__main__':
    # random.seed(2021)
    inputs = read()
    solve(*inputs)
