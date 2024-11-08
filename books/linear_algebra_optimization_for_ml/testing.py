import numpy as np


"""
E = np.array([
    [1, 0, 0],
    [2, 1, 0],
    [0, 0, 1]
])

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

R = np.dot(E, A),

print(E)
print(A)
print(R)
"""

"""
E = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0]
])

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

R = np.dot(E, A),

print(E)
print(A)
print(R)
"""

"""
E = np.array([
    [1, 0],
    [0, -1]
])

A = np.array([1, 2])

R = np.dot(E, A.T),

print(E)
print(A)
print(R)



E = np.array([
    [1, 0],
    [0, -1]
])

A = np.array([1, 2])

R = np.dot(A, E),

print(E)
print(A)
print(R)

"""

"""
x = np.array([1, 3])

A = np.array([
    [0, 1],
    [1, 0]
])


R = np.dot(x.T, A)

R = np.dot(R,x)

print(x)
print(A)
print(R)
"""


"""
P = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]s
])

R = np.matmul(P, P)

R = np.matmul(R, P)
R = np.matmul(R, P)
R = np.matmul(R, P)
R = np.matmul(R, P)
R = np.matmul(R, P)

print(P)
print(R)
"""


"""
E = np.array([
    [0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
])

A = np.array([
    [11, 0, 0, 0, 0],
    [0, 22, 0, 0, 0],
    [0, 0, 33, 0, 0],
    [0,0, 0, 44, 0],
    [0, 0, 0, 0, 55]
])

R = np.dot(np.dot(E.T, A), E),

print(E)
print(A)
print(R)
"""

"""
c1 = np.array([1, 1])
c2 = np.array([1, 1])

R = np.outer(c1, c2)

print(R)
"""

E = np.array([
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, 3]
])

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

R = np.dot(A, E),

print(E)
print(A)
print(R)


