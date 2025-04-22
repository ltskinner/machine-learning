
import numpy as np


# Coefficients of F'(x) = 4x^3 -24x^2 +42x -22
coefficients = [4, -24, 42, -22]

# Find the roots
roots = np.roots(coefficients)

def f_base(x):
    return (x - 1)*(x - 1)*((x - 3) * (x - 3) - 1)

def f_2nd(x):
    return 12*x*x - 48*x + 42

global_min_root = roots[0]
global_min = +10000000000

global_max_root = roots[0]
global_max = -10000000000
local_mins = []
local_maxs = []

for root in roots:
    print(root)

    f2 = f_2nd(root)


    f = f_base(root)

    if f2 > 0:
        local_mins.append(root)
        if f < global_min:
            global_min = f
            global_min_root = root
    elif f2 < 0:
        local_maxs.append(root)
        if f > global_max:
            global_max = f
            global_max_root = root
    else:
        print(f'f2 of {root} = 0')




print(f'local mins:', local_mins)
print(f'global min:', global_min)
print(f'global min_root', global_min_root)

print(f'local maxs:', local_maxs)
print(f'global max:', global_max)
print(f'global max root:', global_max_root)

