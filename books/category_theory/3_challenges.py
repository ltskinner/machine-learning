"""
2.a - partial order

2.b - partial order
    - functionally partial bc equivalent
    - "is a subtype IF"
        - NOT, is a subtype

parent T2
T2 inherits from T1, T1 is a subclass of T2

If it gets a T2 it no break
If it gets T1 it no break
"""

"""
3.

Bool = {True, False}

Show it forms two monoids (set-theoretical)
- && (AND)
- || (OR)

- All thats required from this operation is that it is associative
- and there is one special element that behaves like a unit with respect to it


(a && b) && c = a && (b && c)
(a || b) || c = a || (b || c)

a && N = a  -- true and true == true
N && a = a  -- true and true == true

a || N = a  -- true or false = true
N || a = a  -- false or true = true

"""

"""
4.

object:
    - Bool
    - the N is silent
morphisms:
    - id
    - && true
    - && false

id = (& True)
id . (& True) = (& True)
id . (& False) = (& False)

(& True) = id
(& True) . (& True) = (& True)

(& False) . (& False) = (& False)
(& False) . (& True) = (& False)

"""

# 5.


def mod3(v):
    return v % 3


for i in range(0, 5):
    print(mod3(i))



"""
A   0 = n/3
B   1 = n/3 + 1
C   2 = n/3 + 2

B   1 = 0 + 1
C   2 = 0 + 2

A . B = B
B . B = C
B . C = A



"""
