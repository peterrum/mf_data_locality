#ifndef solver_cg_h
#define solver_cg_h

#include "../common_code/timer.h"

#define SOLVER_CG_WITH_DETAIL

#ifndef SOLVER_CG_WITH_DETAIL

#  include <deal.II/lac/solver_cg.h>

template <typename VectorType>
class SolverCG : public dealii::SolverCG<VectorType>
{
public:
  SolverCG(dealii::SolverControl &cn)
    : dealii::SolverCG<VectorType>(cn)
  {}

  const std::vector<double> &
  get_profile()
  {
    return times;
  }

private:
  std::vector<double> times;
};

#else

template <typename VectorType>
class SolverCG : public dealii::SolverBase<VectorType>
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = dealii::types::global_dof_index;

  /**
   * Constructor.
   */
  SolverCG(dealii::SolverControl &cn)
    : dealii::SolverBase<VectorType>(cn)
    , times(4, 0.0)
  {}


  /**
   * Virtual destructor.
   */
  virtual ~SolverCG() override = default;

  /**
   * Solve the linear system $Ax=b$ for x.
   */
  template <typename MatrixType, typename PreconditionerType>
  void
  solve(const MatrixType &        A,
        VectorType &              x,
        const VectorType &        b,
        const PreconditionerType &preconditioner)
  {
    dealii::SolverControl::State conv = dealii::SolverControl::iterate;
    using number                      = typename VectorType::value_type;

    // Memory allocation
    typename dealii::VectorMemory<VectorType>::Pointer g_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer d_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer h_pointer(this->memory);

    // define some aliases for simpler access
    VectorType &g = *g_pointer;
    VectorType &d = *d_pointer;
    VectorType &h = *h_pointer;

    int    it  = 0;
    double res = -std::numeric_limits<double>::max();

    // resize the vectors, but do not set
    // the values since they'd be overwritten
    // soon anyway.
    g.reinit(x, true);
    d.reinit(x, true);
    h.reinit(x, true);

    number gh, beta;

    // compute residual. if vector is zero, then short-circuit the full
    // computation
    if (!x.all_zero())
      {
        A.vmult(g, x);
        g.add(-1., b);
      }
    else
      {
        ScopedTimer timer(times[1]);
        g.equ(-1., b);
      }
    {
      ScopedTimer timer(times[2]);
      res = g.l2_norm();
    }

    conv = this->iteration_status(0, res, x);
    if (conv == dealii::SolverControl::iterate &&
        std::is_same<PreconditionerType, dealii::PreconditionIdentity>::value == false)
      {
        {
          ScopedTimer timer(times[3]);
          preconditioner.vmult(h, g);
        }
        {
          ScopedTimer timer(times[1]);
          d.equ(-1., h);
        }

        {
          ScopedTimer timer(times[2]);
          gh = g * h;
        }
      }
    else if (conv == dealii::SolverControl::iterate)
      {
        ScopedTimer timer(times[1]);
        d.equ(-1., g);
        gh = res * res;
      }

    while (conv == dealii::SolverControl::iterate)
      {
        it++;
        {
          ScopedTimer timer(times[0]);
          A.vmult(h, d);
        }
        number alpha = 0;
        {
          ScopedTimer timer(times[2]);
          alpha = h * d;
        }

        Assert(std::abs(alpha) != 0., dealii::ExcDivideByZero());
        alpha = gh / alpha;

        {
          ScopedTimer timer(times[1]);
          x.add(alpha, d);
          g.add(alpha, h);
        }
        {
          ScopedTimer timer(times[2]);
          res = g.l2_norm();
        }

        conv = this->iteration_status(it, res, x);

        if (conv != SolverControl::iterate)
          break;

        if (std::is_same<PreconditionerType, dealii::PreconditionIdentity>::value == false)
          {
            {
              ScopedTimer timer(times[3]);
              preconditioner.vmult(h, g);
            }

            beta = gh;
            Assert(std::abs(beta) != 0., ExcDivideByZero());
            {
              ScopedTimer timer(times[2]);
              gh = g * h;
            }
            beta = gh / beta;
            {
              ScopedTimer timer(times[1]);
              d.sadd(beta, -1., h);
            }
          }
        else
          {
            ScopedTimer timer(times[1]);
            beta = gh;
            gh   = res * res;
            beta = gh / beta;
            d.sadd(beta, -1., g);
          }
      }
  }

  const std::vector<double> &
  get_profile()
  {
    return times;
  }

private:
  std::vector<double> times;
};

#endif


#endif
