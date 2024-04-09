


print('1. Implement the identity function')

def identity(x):
    return x

print('2. Implement the composition function')

def composition(f, g, a):
    # wait wtf, we can return the function
    # we could return g(f) but theres no way to get the operators () to f
    # is this an illustrative question??
    return g(f(a))


def x_plus_3(x):
    return x + 3





values = [
    1,
    .1,
]

for value in values:
    result = composition(x_plus_3, identity, value)
    print(value, result)
    assert result == x_plus_3(value)

    result = composition(identity, x_plus_3, value)
    print(value, result)
    assert result == x_plus_3(value)


# -- f after id
assert x_plus_3(identity) == x_plus_3

# -- id after f == f
assert identity(x_plus_3) == x_plus_3


"""
Question 4:
Is the world-wide web a category in any sense?
I want to say you can traverse the www like they are categories,
but you do not have to use compositions to traverse.
I also dont know what an identity function would look like

Are links morphisms?
Yes, some forms


Question 5:
Is Facebook a category, with people as objects and friendships as morphisms?
Yes I would say so. Pages could also be objects, membership as morphisms

I think the bidirectional nature of the friendship makes it an identity

Question 6:
When is a directed graph a category?
I dont think a native directed graph is a category.
Like by default the nodes are all of some type and unless the edges are defined
to be a sort of function (morphism, map, arrow) then theres no change happening.
There also probably needs to be some way to go back from one edge to the previous

Is it safe to say that a category as a directed graph cannot be acyclic?
"""

