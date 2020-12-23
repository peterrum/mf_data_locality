# Data locality of conjugate gradient solvers with matrix-free finite element operators

This project provides various flavors of conjugate gradient solvers to
efficiently implement the ceed benchmark case BP4
http://ceed.exascaleproject.org/bps with the matrix-free evaluation routines
provided by the deal.II finite element library,
https://github.com/dealii/dealii

The project inherits many files from the repository
https://github.com/kronbichler/ceed_benchmarks_dealii
but specializes on different conjugate gradient solvers with or without
preconditioners.
