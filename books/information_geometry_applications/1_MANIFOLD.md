# Chapter 1: Manifold, Divergence and Dually Flat Strucutre

The chapter begins with a manifold and a coordinate system within it. Then, a divergence between two points is defined

- Divergence:
  - representes a degree of separation of two points
  - but it is not a sistance since it is not symmetric with respect to the two points
  Here is the origin of `dually coupled asymmetry`, leading us to a dual world

When a divergence is derived from a convex function in the form of a bregman divergence, two affine structures are induced in the manifold. They are dually coupled via the Legendre transformation. Thus, a convex function provides a maniforld with a dually flat affine structure in addition to a Riemannian metric derived from it. The dually flat structure plays a pivotal role in information geometry, as is shown in the generalized Pythagorean theorem. The dually flat structure is a special case of Riemannian geometry equipped with non-flat dual affine connections, which will be studied in Part II.

## 1.1 Manifolds

### 1.1.1 Manifold Coordinate Systems

An n-dimensional manifold M is a set of points such that each point has n-dimensional extensions in its neighborhood. That is, such a neighborhood is topologically equivalent to an n-dimensional Euclidian space

Intuitively speaking, a manifold is a deformed Euclidian space, like a curved surface in the two-dimensional case. But it may have a different global topology. A sphere is an example which is locally equivalent to a two dimensional Euclidian space, but is curved and has a different global topology because it is compact (bounded and closed)

Locally, a manifold M is equivalent to an n-dimensional Euclidian space E_n, which allows a local coordinate system:

```math
\boldsymbol{\xi} = (\xi_{1}, ..., \xi_{n})
```

Which is composed of n components, such that each point is uniquely specified by its coodinates $`\xi`$ in a neighborhood

Since a manifold may have a topology different from a Euclidian space, in general we need **more than one coordinate neighborhood and coordinate system** to cover all the points of a manifold

The coordinate sysetm is not unique even in a coordinate neighborhood, and there are many coordinate systems. Let $`\zeta = (\zeta_{1}, ..., \zeta_{n})`$ be another coordinate sysetm. When a point $`P \in M`$ is represented in two coordinate systems $`\xi`$ and $`\zeta`$, there is a one-to-one correspondence betwen them and we have relations:

```math
\xi = f(\zeta_{1}, ..., \zeta_{n})
```

```math
\zeta = f(\xi_{1}, ..., \xi_{n})
```

Where $`f`$ and $`f^{-1}`$ are mutually inverse vector-valued functions. They are a coordinate transformation and its inverse transformation. We usually assume that these are differentiable functions of n coordinate variables
