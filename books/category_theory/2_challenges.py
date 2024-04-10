

import random


the_memo = {}



def memoize(f, value):
    """Define a higher-order funciton (or a function object)
    memoize
    takes:
        f (a pure function)
    returns:
        function almost like f
            except
            only calls the original function once for every argument
            stores the results internally
            subsequently returns stored result
            when called with same argument
    """
    global the_memo
    if value not in the_memo.keys():
        the_memo[value] = f(value)
    else:
        print('getting memo')
    
    return the_memo[value]


def x_plus_3(value):
    return value + 3


def get_random_value(seed):
    random.seed(seed)
    return random.randint(0, 3)




for idk in range(10):
    """
    print(f'----------- {seed} ---------------')
    value = get_random_value(seed)
    print(value)
    result = memoize(x_plus_3, value)
    print(result)
    """

    value = random.randint(0, 3)
    result = memoize(get_random_value, value)
    print(result)



"""
*pure functions*: functions that always
produce the same result given the same input and
have no side effects
4.a - factorial - yes
4.b - std::getchar() - no
4.c - i mean it wants to be pure but std cout is not guaranteed
4.d - no


5. How many different functions are there from Bool to Bool
Can you implement them all?

Uhh i mean infinite implementations but only one Type interface

6. Draw a pucture of a category whose only objects are the types:
- Void
- () (unit)
- Bool

With arrows corresponding to all possible funcitons
between these types
Label the arrows with the names of the functions

"""


