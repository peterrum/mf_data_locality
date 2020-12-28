
#include "../common_code/benchmark_test.h"
#include "../common_code/solver_cg.h"

template <typename Operator, typename Preconditioner>
unsigned int
run_cg_solver(const Operator &                                  laplace_operator,
              LinearAlgebra::distributed::Vector<double> &      x,
              const LinearAlgebra::distributed::Vector<double> &b,
              const Preconditioner &)
{
  ReductionControl                                       solver_control(100, 1e-15, 1e-8);
  ::SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

  try
    {
      solver.solve(laplace_operator, x, b, PreconditionIdentity());
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
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  run();
  return 0;
}
