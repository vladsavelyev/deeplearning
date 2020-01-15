import math


def f(v1):
    """
    а(-2) = 3.5
    f(-1) = 0
    f(0)  = -1.5
    f(1)  = -2   <- min
    f(2)  = -1.5
    f(3)  = 0
    f(4)  = 3.5
    """
    return 1/2 * (v1 - 1) ** 2 - 2


def df(v1):
    return v1 - 1


def find_min():
    v1 = 5
    anti_friction = 0.1
    step_n = 0
    while step_n < 100:
        step_n += 1
        # dv = 1 / math.sqrt(df(v1) + 1)
        # new_v1 = v1 + dv
        new_v1 = v1 - df(v1) * anti_friction
        if step_n % 5 == 0:
            print("step:", step_n, "v1:", v1, "new_v1:", new_v1,
                  "f(v1):", f(v1), "f(new_v1):", f(new_v1), "df:", f(new_v1) - f(v1),
                  "diff:", math.fabs(f(v1) - f(new_v1)), "df(v1):", df(v1))
        if math.fabs(f(v1) - f(new_v1)) < 0.00001:
            print("step:", step_n, "v1:", v1, "new_v1:", new_v1,
                  "f(v1):", f(v1), "f(new_v1):", f(new_v1), "df:", f(new_v1) - f(v1),
                  "diff:", math.fabs(f(v1) - f(new_v1)), "df(v1):", df(v1))
            return new_v1, f(new_v1), step_n
        v1 = new_v1


find_min()


