#ifndef solver_cg_optimized_h
#define solver_cg_optimized_h

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>

#include "diagonal_matrix_blocked.h"



template <int n_components, typename Number>
void
do_cg_update3b(const unsigned int                                     start,
               const unsigned int                                     end,
               const Number *                                         r,
               const Number *                                         d,
               const Number *                                         h,
               const Number *                                         prec,
               dealii::Tensor<1, 7, dealii::VectorizedArray<Number>> &sums)
{
  const dealii::VectorizedArray<Number> *arr_r =
    reinterpret_cast<const dealii::VectorizedArray<Number> *>(r);
  const dealii::VectorizedArray<Number> *arr_d =
    reinterpret_cast<const dealii::VectorizedArray<Number> *>(d);
  const dealii::VectorizedArray<Number> *arr_h =
    reinterpret_cast<const dealii::VectorizedArray<Number> *>(h);
  dealii::Tensor<1, 7, dealii::VectorizedArray<Number>> local_sum;
  for (unsigned int i = start / dealii::VectorizedArray<Number>::size();
       i < end / dealii::VectorizedArray<Number>::size();
       ++i)
    {
      dealii::VectorizedArray<Number> arr_prec;
      for (unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
        arr_prec[v] = prec[(i * dealii::VectorizedArray<Number>::size() + v) / n_components];

      local_sum[0] += arr_d[i] * arr_h[i];
      local_sum[1] += arr_h[i] * arr_h[i];
      local_sum[2] += arr_r[i] * arr_h[i];
      local_sum[3] += arr_r[i] * arr_r[i];
      local_sum[6] += arr_r[i] * arr_prec * arr_r[i];
      const dealii::VectorizedArray<Number> zi = arr_prec * arr_h[i];
      local_sum[4] += arr_r[i] * zi;
      local_sum[5] += arr_h[i] * zi;
    }
  for (unsigned int i =
         (end / dealii::VectorizedArray<Number>::size()) * dealii::VectorizedArray<Number>::size();
       i < end;
       ++i)
    {
      local_sum[0][0] += d[i] * h[i];
      local_sum[1][0] += h[i] * h[i];
      local_sum[2][0] += r[i] * h[i];
      local_sum[3][0] += r[i] * r[i];
      local_sum[6][0] += r[i] * prec[i / n_components] * r[i];
      const Number zi = prec[i / n_components] * h[i];
      local_sum[4][0] += r[i] * zi;
      local_sum[5][0] += h[i] * zi;
    }
  sums += local_sum;
}



template <int n_components, typename Number, bool do_update_h>
void
do_cg_update4b(const unsigned int start,
               const unsigned int end,
               Number *           h,
               Number *           x,
               Number *           r,
               Number *           p,
               const Number *     prec,
               const Number       alpha,
               const Number       beta,
               const Number       alpha_old,
               const Number       beta_old)
{
  dealii::VectorizedArray<Number> *arr_p = reinterpret_cast<dealii::VectorizedArray<Number> *>(p);
  dealii::VectorizedArray<Number> *arr_x = reinterpret_cast<dealii::VectorizedArray<Number> *>(x);
  dealii::VectorizedArray<Number> *arr_r = reinterpret_cast<dealii::VectorizedArray<Number> *>(r);
  dealii::VectorizedArray<Number> *arr_h = reinterpret_cast<dealii::VectorizedArray<Number> *>(h);

  if (alpha == Number())
    {
      for (unsigned int i = start / dealii::VectorizedArray<Number>::size();
           i < end / dealii::VectorizedArray<Number>::size();
           ++i)
        {
          dealii::VectorizedArray<Number> arr_prec;
          for (unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
            arr_prec[v] = prec[(i * dealii::VectorizedArray<Number>::size() + v) / n_components];
          arr_p[i] = -arr_prec * arr_r[i];
          if (do_update_h)
            arr_h[i] = dealii::VectorizedArray<Number>();
        }
      for (unsigned int i = (end / dealii::VectorizedArray<Number>::size()) *
                            dealii::VectorizedArray<Number>::size();
           i < end;
           ++i)
        {
          p[i] = -prec[i / n_components] * r[i];
          if (do_update_h)
            h[i] = 0.;
        }
    }
  else if (alpha_old == Number())
    {
      for (unsigned int i = start / dealii::VectorizedArray<Number>::size();
           i < end / dealii::VectorizedArray<Number>::size();
           ++i)
        {
          dealii::VectorizedArray<Number> arr_prec;
          for (unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
            arr_prec[v] = prec[(i * dealii::VectorizedArray<Number>::size() + v) / n_components];
          arr_r[i] += alpha * arr_h[i];
          arr_p[i] = beta * arr_p[i] - arr_prec * arr_r[i];
          if (do_update_h)
            arr_h[i] = dealii::VectorizedArray<Number>();
        }
      for (unsigned int i = (end / dealii::VectorizedArray<Number>::size()) *
                            dealii::VectorizedArray<Number>::size();
           i < end;
           ++i)
        {
          r[i] += alpha * h[i];
          p[i] = beta * p[i] - prec[i / n_components] * r[i];
          if (do_update_h)
            h[i] = 0.;
        }
    }
  else
    {
      const Number alpha_plus_alpha_old = alpha + alpha_old / beta_old;
      const Number alpha_old_beta_old   = alpha_old / beta_old;
      for (unsigned int i = start / dealii::VectorizedArray<Number>::size();
           i < end / dealii::VectorizedArray<Number>::size();
           ++i)
        {
          dealii::VectorizedArray<Number> arr_prec;
          for (unsigned int v = 0; v < dealii::VectorizedArray<Number>::size(); ++v)
            arr_prec[v] = prec[(i * dealii::VectorizedArray<Number>::size() + v) / n_components];
          arr_x[i] += alpha_plus_alpha_old * arr_p[i] + alpha_old_beta_old * arr_prec * arr_r[i];
          arr_r[i] += alpha * arr_h[i];
          arr_p[i] = beta * arr_p[i] - arr_prec * arr_r[i];
          if (do_update_h)
            arr_h[i] = dealii::VectorizedArray<Number>();
        }
      for (unsigned int i = (end / dealii::VectorizedArray<Number>::size()) *
                            dealii::VectorizedArray<Number>::size();
           i < end;
           ++i)
        {
          x[i] += alpha_plus_alpha_old * p[i] + alpha_old_beta_old * prec[i / n_components] * r[i];
          r[i] += alpha * h[i];
          p[i] = beta * p[i] - prec[i / n_components] * r[i];
          if (do_update_h)
            h[i] = 0.;
        }
    }
}



template <typename VectorType>
class SolverCGFullMerge : public dealii::SolverBase<VectorType>
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = dealii::types::global_dof_index;

  /**
   * Constructor.
   */
  SolverCGFullMerge(dealii::SolverControl &cn)
    : dealii::SolverBase<VectorType>(cn)
  {}


  /**
   * Virtual destructor.
   */
  virtual ~SolverCGFullMerge() override = default;

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

    int    it       = 0;
    double res_norm = -std::numeric_limits<double>::max();

    // resize the vectors, but do not set the values since they'd be
    // overwritten soon anyway.
    g.reinit(x, true);
    d.reinit(x, true);
    h.reinit(x, true);

    // compute residual. if vector is zero, then short-circuit the full
    // computation
    if (!x.all_zero())
      {
        A.vmult(g, x);
        g.add(-1., b);
      }
    else
      g.equ(-1., b);
    res_norm = g.l2_norm();

    conv = this->iteration_status(0, res_norm, x);
    if (conv != dealii::SolverControl::iterate)
      return;

    number alpha     = 0.;
    number beta      = 0.;
    number alpha_old = 0.;
    number beta_old  = 0.;

    while (conv == dealii::SolverControl::iterate)
      {
        it++;

        const dealii::Tensor<1, 7> results = A.vmult_with_merged_sums(
          x, g, d, h, preconditioner, alpha, beta, it % 2 == 1 ? alpha_old : 0, beta_old);

        alpha_old = alpha;
        beta_old  = beta;

        Assert(std::abs(results[0]) != 0., dealii::ExcDivideByZero());
        alpha = results[6] / results[0];

        res_norm = std::sqrt(results[3] + 2 * alpha * results[2] + alpha * alpha * results[1]);
        conv     = this->iteration_status(it, res_norm, x);
        if (conv != dealii::SolverControl::iterate)
          {
            if (it % 2 == 1)
              x.add(alpha, d);
            else
              {
                constexpr unsigned int           n_components = MatrixType::n_components;
                dealii::VectorizedArray<number> *arr_p =
                  reinterpret_cast<dealii::VectorizedArray<number> *>(d.begin());
                dealii::VectorizedArray<number> *arr_x =
                  reinterpret_cast<dealii::VectorizedArray<number> *>(x.begin());
                dealii::VectorizedArray<number> *arr_r =
                  reinterpret_cast<dealii::VectorizedArray<number> *>(g.begin());
                const number       alpha_plus_alpha_old = alpha + alpha_old / beta_old;
                const number       alpha_old_beta_old   = alpha_old / beta_old;
                const unsigned int end                  = g.local_size();
                for (unsigned int i = 0; i < end / dealii::VectorizedArray<number>::size(); ++i)
                  {
                    dealii::VectorizedArray<number> arr_prec;
                    for (unsigned int v = 0; v < dealii::VectorizedArray<number>::size(); ++v)
                      arr_prec[v] = preconditioner.get_vector().local_element(
                        (i * dealii::VectorizedArray<number>::size() + v) / n_components);
                    arr_x[i] +=
                      alpha_plus_alpha_old * arr_p[i] + alpha_old_beta_old * arr_prec * arr_r[i];
                  }
                for (unsigned int i = (end / dealii::VectorizedArray<number>::size()) *
                                      dealii::VectorizedArray<number>::size();
                     i < end;
                     ++i)
                  x.local_element(i) +=
                    alpha_plus_alpha_old * d.local_element(i) +
                    alpha_old_beta_old *
                      preconditioner.get_vector().local_element(i / n_components) *
                      g.local_element(i);
              }
            break;
          }

        // Polak-Ribiere like formula to update
        // r^{k+1}^T M^{-1} * (alpha h) = alpha (r^k + alpha h)^T M^{-1} h
        const number gh = alpha * (results[4] + alpha * results[5]);
        beta            = gh / results[6];
      }

    // in case of failure: throw exception
    if (conv != dealii::SolverControl::success)
      AssertThrow(false, dealii::SolverControl::NoConvergence(it, res_norm));
    // otherwise exit as normal
  }
};



#endif
