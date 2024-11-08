# Spaces

A `space` is a **set of elements** (`points`, `vectors`, or `functions` depending on the context) equipped with a **structure** that allows for meaningful operations or relationships to be defined between those elements.

Core components:

- 1. has a **set of elements**, which is the collection of points, objects, vectors, numbers, fns, etc
- 2. has **operations** or **structure**
  - such as metrics (distances), inner product (angles), norms, topologies, or any other relations
  - the structure determines what properties or operations are valid in the space

Tyes of **structures** by space:

- Metric Space
  - structure provided by a **distance function** (metric)
- Vector Space
  - defined over vectors with operations like **vector addition** and **scalar multiplication**
- Normed Space
  - a vector space with a **norm** that assigns a length to each element
- Inner Product Space
  - a vector space with an inner product to measure angles and lengths
- Topological Space
  - Structure based on **open sets**, used to define continuity

## Spaces

| Space | Description | Grounded In | Complexity |
| - | - | - | - |
| Hamming Space | Measures Similarity based on the number of differing bits between binary strings | Discrete Metric | 1 |
| Euclidian Space | Represents n-dimensional space with traditional straight-line distance (L2 norm) | Euclidian Geometry | 1 |
| Manhattan Space (Taxicab Geometry) | Uses the L1 norm, where distance is measured as the sum of absolute differences across dimensions | Euclidian Geometry | 2 |
| Metric Space | A set where distances between points are defined by a metric function, general framework for spaces like Euclidian | Metric Spaces | 2 |
| Probability Space | Models probabilistic events with defined sample space, sigma-algebra, and probability measure | Probability Theory | 2 |
| Graph Space | Represents the structure of graphs or networks, with distances defined by graph metrics like shortest paths | Graph Theory | 2 |
| Affine Space | Similar to vector spaces but without a fixed origin, used in geometry and robotics | Affine Geometry | 2 |
| Hilbert Space | Infinite-dimensional generalization of Euclidian space, used in functional analysis and quantum mechanics | Euclidian Geometry | 3 |
| Banach Space | A vector space with a complete norm, generalizing Euclidian space with more flexible norms | Normed Spaces | 3 |
| Topological Space | Focuses on the properties of space preserved through continuous deformations | Topology | 3 |
| Inner Product Space | A vector space equipped with an inner product, used to define angles and lengths | Inner Product Spaces | 3 |
| Function Space | A set of functions that serve as elements, used in optimization and calculus of variations | Functional Analysis | 3 |
| Tensor Space | Composed of tensors, representing multi-dimensional arrays used in deep learning and linear algebra | Linear Algebra | 3 |
| Riemannian Space | Generalizes Euclidian Space by allowing for curved manifolds with local inner products | Riemannian Geometry | 4 |
| Hyperbolic Space | A space with constant negative curvature, used for embedding hierarchical data | Hyperbolic Geometry | 4 |
| Symplectic Space | Describes the phase space in classical mechanics, used in physics and geometry | Symplectic Geometry | 4 |

## Definitions

- `hyperplane`
  - a "flat" subspace that is one dimension lower than the space its in
  - e.g. 2d plane in 3d space, or 4d plane in 5d space
