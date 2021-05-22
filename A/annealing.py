import random
import math

def rect(x1, x2):
    return x1 if 4 <= x1 <= 10 or 22 <= x1 <= 25 or 45 <= x1 <= 46 else 0


def sphere(x1, x2):
    return -(x1**2 + x2**2)


def easom(x1, x2):
    return math.cos(x1) * math.cos(x2) * pow(math.e, -((x1 - math.pi)**2 + (x2 - math.pi)**2))


def annealing(func, n=1000, T0=2e3, T1=6e2, T=1.9):
    # スコアの最大化
    best_y = 0
    x1 = random.uniform(0, 100)
    x2 = random.uniform(0, 100)
    old_y = func(x1, x2)
    best_y = old_y
    for i in range(n):
        Ti = pow(T0, 1 - i/n) * pow(T1, i/n)
        x1 = random.uniform(0, 100)
        x2 = random.uniform(0, 100)
        new_y = func(x1, x2)
        delta = new_y - old_y
        p = 1 if delta >= 0 else pow(math.e, delta / T)
        if random.random() < p:
            # accept
            old_y = new_y
            if old_y > best_y:
                best_y = old_y
                print("%04d, %.4f" % (i, best_y))


annealing(func=easom, n=100000)




