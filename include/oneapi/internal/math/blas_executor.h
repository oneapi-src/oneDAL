/* file: blas_executor.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef __ONEAPI_INTERNAL_MATH_BLAS_EXECUTOR_H__
#define __ONEAPI_INTERNAL_MATH_BLAS_EXECUTOR_H__

/*
//++
//  Executors for BLAS functions
//--
*/

#if (!defined(ONEAPI_DAAL_NO_MKL_GPU_FUNC) && defined(__SYCL_COMPILER_VERSION))
    #include "mkl_blas.h"
#endif

#include "oneapi/internal/types_utils.h"
#include "types.h"
#include "services/internal/error_handling_helpers.h"
#include "reference_gemm.h"
#include "reference_axpy.h"

#include <CL/sycl.hpp>

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{
namespace interface1
{
/**
 * @defgroup oneapi_internal oneAPIInternal
 * \brief Contains classes of SYCL* abstraction layer
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__GEMMEXECUTOR"></a>
 *  \brief Adapter for GEMM routine
 */
class GemmExecutor
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        const math::Transpose transa;
        const math::Transpose transb;
        const size_t m;
        const size_t n;
        const size_t k;
        const double alpha;
        const UniversalBuffer & a_buffer;
        const size_t lda;
        const size_t offsetA;
        const UniversalBuffer & b_buffer;
        const size_t ldb;
        const size_t offsetB;
        const double beta;
        UniversalBuffer & c_buffer;
        const size_t ldc;
        const size_t offsetC;
        services::Status * status;

        explicit Execute(cl::sycl::queue & queue, const math::Transpose transa, const math::Transpose transb, const size_t m, const size_t n,
                         const size_t k, const double alpha, const UniversalBuffer & a_buffer, const size_t lda, const size_t offsetA,
                         const UniversalBuffer & b_buffer, const size_t ldb, const size_t offsetB, const double beta, UniversalBuffer & c_buffer,
                         const size_t ldc, const size_t offsetC, services::Status * status)
            : queue(queue),
              transa(transa),
              transb(transb),
              m(m),
              n(n),
              k(k),
              alpha(alpha),
              a_buffer(a_buffer),
              lda(lda),
              offsetA(offsetA),
              b_buffer(b_buffer),
              ldb(ldb),
              offsetB(offsetB),
              beta(beta),
              c_buffer(c_buffer),
              ldc(ldc),
              offsetC(offsetC),
              status(status)
        {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            auto a_buffer_t = a_buffer.template get<T>();
            auto b_buffer_t = b_buffer.template get<T>();
            auto c_buffer_t = c_buffer.template get<T>();

#ifdef ONEAPI_DAAL_NO_MKL_GPU_FUNC
            ReferenceGemm<T> functor;
#else
            MKLGemm<T> functor(queue);
#endif

            services::Status statusGemm =
                functor(transa, transb, m, n, k, (T)alpha, a_buffer_t, lda, offsetA, b_buffer_t, ldb, offsetB, (T)beta, c_buffer_t, ldc, offsetC);

            services::internal::tryAssignStatus(status, statusGemm);
        }
    };

public:
    static void run(cl::sycl::queue & queue, const math::Transpose transa, const math::Transpose transb, const size_t m, const size_t n,
                    const size_t k, const double alpha, const UniversalBuffer & a_buffer, const size_t lda, const size_t offsetA,
                    const UniversalBuffer & b_buffer, const size_t ldb, const size_t offsetB, const double beta, UniversalBuffer & c_buffer,
                    const size_t ldc, const size_t offsetC, services::Status * status)
    {
        Execute op(queue, transa, transb, m, n, k, alpha, a_buffer, lda, offsetA, b_buffer, ldb, offsetB, beta, c_buffer, ldc, offsetC, status);
        TypeDispatcher::floatDispatch(a_buffer.type(), op);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__SYRKEXECUTOR"></a>
 *  \brief Adapter for SYRK routine
 */
class SyrkExecutor
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        const math::UpLo upper_lower;
        const math::Transpose trans;
        const size_t n;
        const size_t k;
        const double alpha;
        const UniversalBuffer & a_buffer;
        const size_t lda;
        const size_t offsetA;
        const double beta;
        UniversalBuffer & c_buffer;
        const size_t ldc;
        const size_t offsetC;
        services::Status * status;

        explicit Execute(cl::sycl::queue & queue, const math::UpLo upper_lower, const math::Transpose trans, const size_t n, const size_t k,
                         const double alpha, const UniversalBuffer & a_buffer, const size_t lda, const size_t offsetA, const double beta,
                         UniversalBuffer & c_buffer, const size_t ldc, const size_t offsetC, services::Status * status)
            : queue(queue),
              upper_lower(upper_lower),
              trans(trans),
              n(n),
              k(k),
              alpha(alpha),
              a_buffer(a_buffer),
              lda(lda),
              offsetA(offsetA),
              beta(beta),
              c_buffer(c_buffer),
              ldc(ldc),
              offsetC(offsetC),
              status(status)
        {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            auto a_buffer_t = a_buffer.template get<T>();
            auto c_buffer_t = c_buffer.template get<T>();

            const math::Transpose transInv = trans == math::Transpose::NoTrans ? math::Transpose::Trans : math::Transpose::NoTrans;

            services::Status statusSyrk;

#ifdef ONEAPI_DAAL_NO_MKL_GPU_FUNC
            ReferenceGemm<T> functor;
            statusSyrk =
                functor(transInv, trans, k, k, n, (T)alpha, a_buffer_t, lda, offsetA, a_buffer_t, lda, offsetA, (T)beta, c_buffer_t, ldc, offsetC);
#else
            MKLSyrk<T> functor(queue);
            statusSyrk = functor(upper_lower, transInv, k, n, (T)alpha, a_buffer_t, lda, offsetA, (T)beta, c_buffer_t, ldc, offsetC);
#endif

            services::internal::tryAssignStatus(status, statusSyrk);
        }
    };

public:
    static void run(cl::sycl::queue & queue, const math::UpLo upper_lower, const math::Transpose trans, const size_t n, const size_t k,
                    const double alpha, const UniversalBuffer & a_buffer, const size_t lda, const size_t offsetA, const double beta,
                    UniversalBuffer & c_buffer, const size_t ldc, const size_t offsetC, services::Status * status)
    {
        Execute op(queue, upper_lower, trans, n, k, alpha, a_buffer, lda, offsetA, beta, c_buffer, ldc, offsetC, status);
        TypeDispatcher::floatDispatch(a_buffer.type(), op);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__AXPYEXECUTOR"></a>
 *  \brief Adapter for AXPY routine
 */
class AxpyExecutor
{
private:
    struct Execute
    {
        cl::sycl::queue & queue;
        const uint32_t n;
        const double a;
        const UniversalBuffer& x_buffer;
        const int incx;
        const UniversalBuffer& y_buffer;
        const int incy;
        services::Status * status;

        explicit Execute(cl::sycl::queue & queue, const uint32_t n, const double a, const UniversalBuffer& x_buffer, const int incx,
               const UniversalBuffer& y_buffer, const int incy, services::Status * status = NULL)
            : queue(queue),
              n(n),
              a(a),
              x_buffer(x_buffer),
              incx(incx),
              y_buffer(y_buffer),
              incy(incy),
              status(status)
        {}

        template <typename T>
        void operator()(Typelist<T>)
        {
            auto x_buffer_t = x_buffer.template get<T>();
            auto y_buffer_t = y_buffer.template get<T>();

            services::Status statusAxpy;
#ifdef ONEAPI_DAAL_USE_MKL_GPU_FUNC
            MKLAxpy<T> functor(queue);
            statusAxpy = functor(n, (T)a, x_buffer_t, incx, y_buffer_t, incy);
#else
            ReferenceAxpy<T> functor;
            statusAxpy =
                functor(n, (T)a, x_buffer_t, incx, y_buffer_t, incy);
#endif

            services::internal::tryAssignStatus(status, statusAxpy);
        }
    };

public:
    static void run(cl::sycl::queue & queue, const uint32_t n, const double a, const UniversalBuffer x_buffer, const int incx,
                    const UniversalBuffer y_buffer, const int incy, services::Status * status = NULL)
    {
        Execute op(queue, n, a, x_buffer, incx, y_buffer, incy, status);
        TypeDispatcher::floatDispatch(x_buffer.type(), op);
    }
};

/** @} */

} // namespace interface1

using interface1::GemmExecutor;
using interface1::SyrkExecutor;
using interface1::AxpyExecutor;

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
