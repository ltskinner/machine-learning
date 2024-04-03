


print('1. Implement the identity function')

def identity(x):
    return x

print('2. Implement the composition function')

def composition(f, g, a):
    # wait wtf, we can return the function
    # we could return g(f) but theres no way to get the operators () to f
    # is this an illustrative question??
    return g(f(a))



values = [
    1,
    .1,
    None,
    'string',
    True,
]

for value in values:
    assert composition(value, identity) == value
    assert composition(identity, value) == value


