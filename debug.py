

import numpy as np

sq6 = np.sqrt(6)

a2 = np.array([2, 1, 1])
a3 = np.array([3, 3, 2])
q1 = np.array([1/sq6, 2/sq6, 1/sq6])
#q1 = np.array([1, 2, 1])

u2 = a2.T - np.dot(np.dot(q1, a2), q1.T)

print(np.dot(q1, a2))

print('5/6:', 5/6)


print(np.dot(np.dot(q1, a2), q1.T))
print('u2:', u2)

print('sum:', sum([u**2 for u in u2]))
q2 = u2 / np.sqrt(sum([u**2 for u in u2]))
print('q2:', q2)



u3 = a3.T - np.dot(np.dot(q1, a3), q1.T) - np.dot(np.dot(q2, a3), q2.T)
print('u3:', u3)
q3 = u3 / np.sqrt(sum([u**2 for u in u3]))
print('q3:', q3)


Q = np.array([q1, q2, [0, 0, 0]]).T
b = np.array([1, 1, 1]).T

print(Q)
print(b)

print(np.matmul(Q, Q.T))
print(np.dot(Q, Q.T))

b_prime = np.dot(np.dot(Q, Q.T), b)
print(b_prime)
