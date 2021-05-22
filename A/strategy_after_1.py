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


def rect_size(rect):
    (left, top, right, bottom) = rect
    return (right - left) * (bottom - top)


def point_in_rect(x, y, rect):
    (left, top, right, bottom) = rect
    return left <= x < right and top <= y < bottom


def rect_in_field(rect):
    (left, top, right, bottom) = rect
    return 0 <= left and right <= 10000 and 0 <= top and bottom <= 10000


def rect_in_rect(rect0, rect1):
    (left0, top0, right0, bottom0) = rect0
    (left1, top1, right1, bottom1) = rect1
    cx0 = right0 + left0
    cx1 = right1 + left1
    cy0 = bottom0 + top0
    cy1 = bottom1 + top1
    diff_cx = abs(cx1 - cx0)
    diff_cy = abs(cy1 - cy0)
    th_x = (right0 - left0) + (right1 - left1)
    th_y = (bottom0 - top0) + (bottom1 - top1)
    return diff_cx <= th_x and diff_cy <= th_y


def check_adspace(N, ads, i, new_ad):
    if not rect_in_field(new_ad):
        return False
    for j in range(N):
        if i != j and rect_in_rect(ads[j], new_ad):
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


def check_cluster(M, ZWL, i, new_ad):
    for j in range(M):
        z, w, l = ZWL[j]
        if i != l and point_in_rect(z, w, new_ad):
            return False
    return True


def random_dilate(N, XYR, ads, i, T=20, U=5):
    x, y, r = XYR[i]
    (left, top, right, bottom) = ads[i]
    s = rect_size(ads[i])
    ar = r
    for t in range(T):
        # ar = s + (r - s) * math.cos(math.pi * t / (T << 1))
        ar = s + (r - s) * (1 - t / T)
        div = list_divisors(int(ar))
        #random.shuffle(div)
        for width in div[::-1]:
            height = int(ar) // width
            for _ in range(U):
                factor_l = random.random()
                factor_t = random.random()
                left = math.floor(x - width * factor_l)
                top = math.floor(y - height * factor_t)
                new_ad = (left, top, left + width, top + height)
                if point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                    ads[i] = new_ad
                    return


def voronoi_dilate(N, XYR, ads, M=10000):
    ZWL = []
    # seeding
    for i in range(M):
        rx = random.uniform(0, 10000)
        ry = random.uniform(0, 10000)
        ZWL.append([rx, ry, -1])
    # make cluster
    for j in range(M):
        rx, ry, _ = ZWL[j]
        label = -1
        min_dist = 10**9+1
        for i in range(N):
            x, y, r = XYR[i]
            dist = abs(ry - y) + abs(rx - x)
            if min_dist > dist:
                min_dist = dist
                label = i
        ZWL[j][2] = label
    # diagonal dilate
    for i in range(N):
        x, y, r = XYR[i]
        left, top, right, bottom = ads[i]
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left - mid, top - mid, right + mid, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad) and check_cluster(M, ZWL, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            left -= ok
            top -= ok
            right += ok
            bottom += ok
        ads[i] = (left, top, right, bottom)

def get_score(r, s):
    return 1 - (1 - min(r, s) / max(r, s))**2


def get_total_score(N, XYR, ads):
    return 10**9 * sum([get_score(XYR[i][2], rect_size(ads[i])) for i in range(N)]) / N


def direction_decomposition(cx, cy, ox, oy):
    # 点(ox, oy) から見た (cx, cy) の向きを返す
    vx = RIGHT if cx > ox else LEFT
    vy = BOTTOM if cy > oy else TOP
    return vx, vy


def translate(N, XYR, ads, i, direction=-1):
    # [単位操作] 平行移動
    x, y, r = XYR[i]
    left, top, right, bottom = ads[i]
    if direction == -1:
        direction = random.randint(0, 3)
    dx, dy = 0, 0
    if direction == LEFT:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left - mid, top, right - mid, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            dx = -ok
    elif direction == TOP:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left, top - mid, right, bottom - mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            dy = -ok
    elif direction == RIGHT:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left + mid, top, right + mid, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            dx = ok
    else:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left, top + mid, right, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            dy = ok
    return (left + dx, top + dy, right + dx, bottom + dy)


def small_dilate(N, XYR, ads, i, direction=-1, ad_i=None):
    # [単位操作] 膨張
    x, y, r = XYR[i]
    if ad_i is None:
        ad_i = ads[i]
    left, top, right, bottom = ad_i

    if direction == -1:
        direction = random.randint(0, 3)
    if direction == LEFT:
        new_ad = (left - 1, top, right, bottom)
        if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
            return new_ad
    elif direction == TOP:
        new_ad = (left, top - 1, right, bottom)
        if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
            return new_ad
    elif direction == RIGHT:
        new_ad = (left, top, right + 1, bottom)
        if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
            return new_ad
    else:
        new_ad = (left, top, right, bottom + 1)
        if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
           return new_ad
    return ad_i


def dilate(N, XYR, ads, i, direction=-1, ad_i=None):
    # [単位操作] 膨張
    x, y, r = XYR[i]
    if ad_i is None:
        ad_i = ads[i]
    left, top, right, bottom = ad_i

    if direction == -1:
        direction = random.randint(0, 3)
    if direction == LEFT:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left - mid, top, right, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            left -= ok
    elif direction == TOP:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left, top - mid, right, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            top -= ok
    elif direction == RIGHT:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left, top, right + mid, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            right += ok
    else:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            new_ad = (left, top, right, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            bottom += ok
    return (left, top, right, bottom)


def reshape(N, XYR, ads, i, direction=-1, corner=-1):
    # [単位操作] 等積変形
    x, y, r  = XYR[i]
    left, top, right, bottom = ads[i]
    s = rect_size(ads[i])
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
            new_ad = (left - mid, top, right, top + height)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            width = right - left + ok
            height = int(s / width)
            new_ad = (left - ok, top, right, top + height)
    # 2. left を mid だけ増やして top を減らす
    elif direction == LEFT and corner == 1:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            width = right - left + mid
            height = int(s / width)
            new_ad = (left - mid, bottom - height, right, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            width = right - left + ok
            height = int(s / width)
            new_ad = (left - ok, bottom - height, right, bottom)
    # 3. top を mid だけ増やして left を減らす
    elif direction == TOP and corner == 0:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            height = bottom - top + mid
            width = int(s / height)
            new_ad = (right - width, top - mid, right, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            height = bottom - top + ok
            width = int(s / height)
            new_ad = (right - width, top - ok, right, bottom)
    # 4. top を mid だけ増やして right を減らす
    elif direction == TOP and corner == 1:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            height = bottom - top + mid
            width = int(s / height)
            new_ad = (left, top - mid, left + width, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            height = bottom - top + ok
            width = int(s / height)
            new_ad = (left, top - ok, left + width, bottom)
    # 5. right を mid だけ増やして top を減らす
    elif direction == RIGHT and corner == 0:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            width = right - left + mid
            height = int(s / width)
            new_ad = (left, bottom - height, right + mid, bottom)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            width = right - left + ok
            height = int(s / width)
            new_ad = (left, bottom - height, right + ok, bottom)
    # 6. right を mid だけ増やして bottom を減らす
    elif direction == RIGHT and corner == 1:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            width = right - left + mid
            height = int(s / width)
            new_ad = (left, top, right + mid, top + height)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            width = right - left + ok
            height = int(s / width)
            new_ad = (left, top, right + ok, top + height)
    # 7. bottom を mid だけ増やして right を減らす
    elif direction == BOTTOM and corner == 0:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            height = bottom - top + mid
            width = int(s / height)
            new_ad = (left, top, left + width, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            height = bottom - top + ok
            width = int(s / height)
            new_ad = (left, top, left + width, bottom + ok)
    # 8. bottom を mid だけ増やして left を減らす
    else:
        ok, ng = 0, 10001
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            height = bottom - top + mid
            width = int(s / height)
            new_ad = (right - width, top, right, bottom + mid)
            if rect_size(new_ad) <= r and point_in_rect(x, y, new_ad) and check_adspace(N, ads, i, new_ad):
                ok = mid
            else:
                ng = mid
        if ok < 10000:
            height = bottom - top + ok
            width = int(s / height)
            new_ad = (right - width, top, right, bottom + ok)
    return new_ad


def swap(N, XYR, ads, i, direction=-1):
    # [単位操作] 領域の交換
    x, y, r = XYR[i]
    new_ad_i = ads[i]
    left, top, right, bottom = ads[i]
    if direction == -1:
        direction = random.randint(0, 3)
    # 隣接する矩形を選ぶ
    ad_j = None
    if direction == LEFT:
        x1, y1, x2, y2 = left-1, top, left-1, bottom
        x3, y3, x4, y4 = 0, top, 10000, bottom
    elif direction == TOP:
        x1, y1, x2, y2 = left, top-1, right, top-1
        x3, y3, x4, y4 = left, 0, right, 10000
    elif direction == RIGHT:
        x1, y1, x2, y2 = right+1, top, right+1, bottom
        x3, y3, x4, y4 = 0, top, 10000, bottom
    else:
        x1, y1, x2, y2 = left, bottom+1, right, bottom+1
        x3, y3, x4, y4 = left, 0, right, 10000
    j = -1
    new_ad_j = None
    for k in range(N):
        jx, jy, jr = XYR[k]
        ad_j = ads[k]
        if point_in_rect(x1, y1, ad_j) and point_in_rect(x2, y2, ad_j) and not point_in_rect(jx, jy, (x3, y3, x4, y4)):
            j = k
            break
    # 領域を交換する (ただし new_ad_i の面積が r を超えないようにする)
    if j >= 0:
        jx, jy, jr = XYR[j]
        jleft, jtop, jright, jbottom = ads[j]
        if direction == LEFT:
            new_left = min(jleft, right - max(1, math.floor(r / (bottom - top))))
            new_ad_i = (new_left, top, right, bottom)
            if jy < y:
                new_ad_j = (jleft, jtop, jright, top)
            else:
                new_ad_j = (jleft, bottom, jright, jbottom)
        elif direction == TOP:
            new_top = min(jtop, bottom - max(1, math.floor(r / (right - left))))
            new_ad_i = (left, new_top, right, bottom)
            if x < jx:
                new_ad_j = (right, jtop, jright, jbottom)
            else:
                new_ad_j = (jleft, jtop, left, jbottom)
        elif direction == RIGHT:
            new_right = min(jright, left + max(1, math.floor(r / (bottom - top))))
            new_ad_i = (left, top, new_right, bottom)
            if y < jy:
                new_ad_j = (jleft, bottom, jright, jbottom)
            else:
                new_ad_j = (jleft, jtop, jright, top)
        else:
            new_bottom = min(jbottom, top + max(1, math.floor(r / (right - left))))
            new_ad_i = (left, top, right, new_bottom)
            if jx < x:
                new_ad_j = (jleft, jtop, left, jbottom)
            else:
                new_ad_j = (right, jtop, jright, jbottom)
        return new_ad_i, j, new_ad_j
    return new_ad_i, i, new_ad_i


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


def annealing_v1(N, XYR, ads, n=10000, T0=1e6, T1=1e-2, a_th=0.5):
    for t in range(n):
        T = pow(T0, 1 - t/n) * pow(T1, t/n)
        i = random.randint(0, N-1)
        x, y, r = XYR[i]
        old_score = get_score(r, rect_size(ads[i]))
        new_ad_i = ads[i]
        a = random.random()
        # 単位操作を1つ実行する
        if a < a_th:
            new_ad_i = reshape(N, XYR, ads, i)
            # new_ad_i = dilate(N, XYR, ads, i, ad_i=new_ad_i)
            new_ad_i = small_dilate(N, XYR, ads, i, ad_i=new_ad_i)
            new_score = get_score(r, rect_size(new_ad_i))
        else:
            new_ad_i = translate(N, XYR, ads, i)
            # new_ad_i = dilate(N, XYR, ads, i, ad_i=new_ad_i)
            new_ad_i = small_dilate(N, XYR, ads, i, ad_i=new_ad_i)
            new_score = get_score(r, rect_size(new_ad_i))
        delta = new_score - old_score
        q = pow(math.e, delta / T)
        p = 1 if delta >= 0 else q
        if random.random() < p:
            ads[i] = new_ad_i


def annealing_v2(N, XYR, ads, n=10000, T0=1e6, T1=1e-2, target=None):
    if target is None:
        target = list(range(N))
    for t in range(n):
        T = pow(T0, 1 - t/n) * pow(T1, t/n)
        i = random.choice(target)
        x, y, r = XYR[i]
        new_ad_i = ads[i]
        j = i
        new_ad_j = ads[j]
        a = random.random()
        # 単位操作を1つ実行する
        if a < 0.45:
            old_score = get_score(r, rect_size(ads[i]))
            new_ad_i = reshape(N, XYR, ads, i)
            new_ad_i = dilate(N, XYR, ads, i, ad_i=new_ad_i)
            new_score = get_score(r, rect_size(new_ad_i))
        elif a < 0.9:
            old_score = get_score(r, rect_size(ads[i]))
            new_ad_i = translate(N, XYR, ads, i)
            new_ad_i = dilate(N, XYR, ads, i, ad_i=new_ad_i)
            new_score = get_score(r, rect_size(new_ad_i))
        else:
            direction = random.randint(0, 3)
            jx, jy, jr = XYR[j]
            old_score = get_score(r, rect_size(ads[i])) + get_score(jr, rect_size(new_ad_j))
            new_ad_i, j, new_ad_j = swap(N, XYR, ads, i, direction=direction)
            new_ad_i = dilate(N, XYR, ads, i, ad_i=new_ad_i, direction=direction)
            new_score = get_score(r, rect_size(new_ad_i)) + get_score(jr, rect_size(new_ad_j))
        delta = new_score - old_score
        q = pow(math.e, delta / T)
        p = 1 if delta > 0 else q
        if random.random() < p:
            ads[i] = new_ad_i
            ads[j] = new_ad_j


def init_ads(N, XYR):
    ads = []
    for i in range(N):
        x, y, r = XYR[i]
        ads.append((x, y, x+1, y+1))
    return ads


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def step1(N, XYR):
    # 1.1. set 1px ads
    ads = init_ads(N, XYR)
    return ads


def step2(N, XYR, ads, n):
    annealing_v1(N, XYR, ads, n=n)
    logger.info("step 2: {:.06f}".format(get_total_score(N, XYR, ads)))
    return ads


def step3(N, XYR, ads):
    # 3. dilate
    for i in range(N):
        for d in random.sample(range(4), k=4):
            ads[i] = dilate(N, XYR, ads, i, direction=d)
    logger.info("step 3: {:.06f}".format(get_total_score(N, XYR, ads)))
    return ads


def solve(N, XYR):
    best_total = 0
    best_ads = None

    for i in range(5):
        ads = step1(N, XYR)
        total = get_total_score(N, XYR, ads)
        logger.info("step 1: {:.06f}".format(total))
        if best_total < total:
            best_ads = ads
            best_total = total
    
    
    best_ads = step2(N, XYR, best_ads, n=45000)
    best_total = get_total_score(N, XYR, best_ads)
    best_ads = step3(N, XYR, best_ads)
    best_total = get_total_score(N, XYR, best_ads)
    
    for i in range(N):
        print(*best_ads[i])
    logger.info("best_total: {:.06f}".format(best_total))


if __name__ == '__main__':
    random.seed(2021)
    inputs = read()
    solve(*inputs)
