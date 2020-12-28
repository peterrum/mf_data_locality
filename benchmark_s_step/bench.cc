
#include "../common_code/benchmark.h"
#include "../common_code/solver_cg_s_step.h"

#define FORCE_ITERATION

unsigned int n_steps = 1;

template <typename Operator, typename Preconditioner>
std::pair<unsigned int, std::vector<double>>
run_cg_solver(const Operator &                                  laplace_operator,
              LinearAlgebra::distributed::Vector<double> &      x,
              const LinearAlgebra::distributed::Vector<double> &b,
              const Preconditioner &                            preconditioner)
{
  (void)preconditioner;

  ReductionControl solver_control(100 / n_steps, 1e-15, 1e-8);
  SolverCGSStep    solver(solver_control, n_steps);

  try
    {
      solver.solve(laplace_operator, x, b);
    }
  catch (SolverControl::NoConvergence &e)
    {
      // prevent the solver to throw an exception in case we should need more
      // than 100 iterations
    }

  return {solver_control.last_step() * n_steps, solver.get_profile()};
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 1, ExcNotImplemented());

  n_steps = std::atoi(argv[1]);

  run(argc - 1, argv + 1);
  return 0;
}
