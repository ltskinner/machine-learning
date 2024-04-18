
from typing import Tuple
from typing import Callable as fn


logger = ''


def make_pair(a, b):
    return (a, b)  # alias for tuple


"""
def negate(b: bool) -> bool:
    logger += "Not so! "
    return not b
"""
"""
def negate(b: bool) -> Tuple[bool, str] | None:
    return make_pair(not b, "Not so! ")

def toUpper(s: str) -> str:
    return 
"""



pair = (bool, str)

def isEven(n: int) -> Tuple[bool, str]:
    result = n % 2 == 0
    return make_pair(result, 'isEven ')  # why dont we make this a decorator?

def negate(b: bool) -> Tuple[bool, str]:
    return make_pair(not b, "Not so! ")

def isOdd(n: int) -> Tuple[bool, str]:
    v1, log1 = isEven(n)
    v2, log2 = negate(v1)
    return make_pair(v2, log1 + log2)


# !! not builtin

#def compose(m1: fn, m2: fn) -> 

#compose: fn[[bool, str], Tuple[bool, str]] = lambda m1, m2 : m1

# this does work just not with tuple returns
#compose = lambda m1, m2: (lambda x: m2(m1(x)))
# like I get what were trying to do, the problem is the return types
# I feel like I get the tuple return structure well
# and the question is whether I can abuse notation in python to achieve this
# or whether its not possible



compose = lambda m1, m2: (lambda x: m1(x)[0], m1(x)[0])
compose = lambda m1, m2: (
    lambda x: (
        
        lambda p1_first, p1_second: m2(p1_first)
    )
)

#lambda p1_first, p1_second: m2(m1(x))

"""
p1 = m1(x)
p2 = m2(p1[0])

return p2[0], p1[1]+p2[1]
"""


compose = lambda m1=(lambda x: m1(x)), m2=(lambda p2: [p2[0], p2[1]]): (
    lambda x:
    m1, m2
)



def f1(p1):
    print(p1)
    return p1*3

def f2(p2):
    print(p2)
    return p2*2

res = compose(f1, f2)(1)
print(res)


