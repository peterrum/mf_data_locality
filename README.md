# Enhancing data locality of the conjugate gradient method for high-order matrix-free finite-element  implementations

This project provides various flavors of conjugate gradient solvers to
efficiently implement the [CEED benchmark case BP4](http://ceed.exascaleproject.org/bps) 
with the matrix-free evaluation routines
provided by the [deal.II finite element library](https://github.com/dealii/dealii) on CPUs. The algorithms and 
their efficient implementation are discussed int the manuscript:

```
@article{kronbichler2021,
  title         = {Enhancing data locality of the conjugate gradient method for high-order 
                   matrix-free finite-element implementations},
  author        = {Kronbichler, Martin and Sashko, Dmytro and Munch, Peter},
  year          = {2021},
  eprint        = {},
  archivePrefix = {arXiv},
  primaryClass  = {}
}
```

The manuscript proposes a novel version of the conjugate 
gradient solver [without](benchmark_merged) and [with preconditioners](benchmark_precond_merged)
for cheap preconditioners. The algorithm relies on a data-dependency analysis and interleaves the vector
updates and inner products in a CG iteration with the matrix-vector product.


In related projects, the propsed algorithms are applied to
[CEED benachmarks BP1-BP6](https://github.com/kronbichler/ceed_benchmarks_dealii) and to 
[BP5 on GPUs](https://github.com/kronbichler/ceed_benchmarks_dealii).

