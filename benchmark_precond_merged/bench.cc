
#include "../common_code/benchmark.h"

template <typename Operator, typename Preconditioner>
unsigned int
run_cg_solver(const Operator &                                  laplace_operator,
              LinearAlgebra::distributed::Vector<double> &      x,
              const LinearAlgebra::distributed::Vector<double> &b,
              const Preconditioner &                            preconditioner)
{
  ReductionControl                                              solver_control(100, 1e-15, 1e-8);
  SolverCGFullMerge<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

  try
    {
      solver.solve(laplace_operator, x, b, preconditioner);
      return solver_control.last_step();
    }
  catch (SolverControl::NoConvergence &e)
    {
      // prevent the solver to throw an exception in case we should need more
      // than 100 iterations
      return solver_control.last_step();
    }
}



int
main(int argc, char **argv)
{
  run(argc, argv);
  return 0;
}
