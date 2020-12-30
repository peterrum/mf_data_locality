
#include "../common_code/benchmark_test.h"
#include "../common_code/solver_cg_s_step.h"

unsigned int n_steps = 1;

template <typename Operator, typename Preconditioner>
unsigned int
run_cg_solver(const Operator &                                  laplace_operator,
              LinearAlgebra::distributed::Vector<double> &      x,
              const LinearAlgebra::distributed::Vector<double> &b,
              const Preconditioner &                            preconditioner)
{
  (void)preconditioner;

  ReductionControl solver_control(100 / n_steps, 1e-15, 1e-8);

  try
    {
      if (n_steps == 1)
        SolverCGSStep<1>(solver_control).solve(laplace_operator, x, b);
      else if (n_steps == 2)
        SolverCGSStep<2>(solver_control).solve(laplace_operator, x, b);
      else if (n_steps == 4)
        SolverCGSStep<4>(solver_control).solve(laplace_operator, x, b);
      else if (n_steps == 6)
        SolverCGSStep<6>(solver_control).solve(laplace_operator, x, b);
      else
        Assert(false, ExcNotImplemented());

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

  n_steps = 1;
  run(5);

  n_steps = 2;
  run(5);

  n_steps = 4;
  run(5);

  return 0;
}
