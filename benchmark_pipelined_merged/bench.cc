#include "../common_code/benchmark.h"
#include "../common_code/solver_cg_pipelined_merged.h"

template <typename Operator, typename Preconditioner>
std::pair<unsigned int, std::vector<double>>
run_cg_solver(const Operator &                                  laplace_operator,
              LinearAlgebra::distributed::Vector<double> &      x,
              const LinearAlgebra::distributed::Vector<double> &b,
              const Preconditioner &)
{
  ReductionControl solver_control(100, 1e-15, 1e-8);
  SolverCGPipelinedMerged<LinearAlgebra::distributed::Vector<double>> solver(solver_control);

  try
    {
      solver.solve(laplace_operator, x, b, PreconditionIdentity());
    }
  catch (SolverControl::NoConvergence &e)
    {
      // prevent the solver to throw an exception in case we should need more
      // than 100 iterations
    }

  return {solver_control.last_step(), std::vector<double>()};
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  run(argc, argv);
  return 0;
}
