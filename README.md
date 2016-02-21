# pyMMMF
Maximum-Margin Matrix Factorization in Python

## What does this code do?
This is an implementation of Maximum-Marging Matrix Factorization (which can be used for Matrix Completion).
It is based on "Maximum-Margin Matrix Factorization" / Srebro|Rennie|Jaakkola and the original matlab(+yalmip) code is available [here](http://ttic.uchicago.edu/~nati/mmmf/code.html).

This version, opposite to the original code only implements solving of the primal optimization problem, which is only possible for toy-instances.

There are two different versions in this repo: one is based on cvxpy, the other one is based on picos. Both are incredible libs for working with
convex optimization problems!

## Is there an alternative for large datasets / real-world applications?
Because the optimization problem is convex (with some mild assumptions), one could use Stochastic Gradient Descent. I implemented this [here](https://github.com/sschnug/MaxNormRegCollaborativeFiltering_SGD) (again: based on some academic paper).
This approach is fast and can be used on the Netflix dataset (collaborative filtering; sparse matrix with 100M entries; not publicly available).
