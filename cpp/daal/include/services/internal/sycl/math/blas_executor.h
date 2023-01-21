/* file: blas_executor.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include <sycl/sycl.hpp>

#if (!defined(ONEAPI_DAAL_NO_MKL_GPU_FUNC) && defined(__SYCL_COMPILER_VERSION))
    #include "services/internal/sycl/math/mkl_blas.h"
#endif

#include "services/internal/sycl/types_utils.h"
#include "services/internal/sycl/math/types.h"
#include "services/internal/sycl/math/reference_gemm.h"
#include "services/internal/sycl/math/reference_axpy.h"

namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
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
        ::sycl::queue & queue;
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

        explicit Execute(::sycl::queue & queue, const math::Transpose transa, const math::Transpose transb, const size_t m, const size_t n,
                         const size_t k, const double alpha, const UniversalBuffer & a_buffer, const size_t lda, const size_t offsetA,
                         const UniversalBuffer & b_buffer, const size_t ldb, const size_t offsetB, const double beta, UniversalBuffer & c_buffer,
                         const size_t ldc, const size_t offsetC)
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
              offsetC(offsetC)
        {}

        size_t getAExpectedSize() const { return (transa == math::Transpose::NoTrans) ? lda * k : lda * m; }

        size_t getBExpectedSize() const { return (transb == math::Transpose::NoTrans) ? ldb * n : ldb * k; }

        size_t getCExpectedSize() const { return ldc * n; }

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER(a_buffer, T, getAExpectedSize());
            DAAL_ASSERT_UNIVERSAL_BUFFER(b_buffer, T, getBExpectedSize());
            DAAL_ASSERT_UNIVERSAL_BUFFER(c_buffer, T, getCExpectedSize());

            auto a_buffer_t = a_buffer.template get<T>();
            auto b_buffer_t = b_buffer.template get<T>();
            auto c_buffer_t = c_buffer.template get<T>();

#ifdef ONEAPI_DAAL_NO_MKL_GPU_FUNC
            ReferenceGemm<T> functor;
#else
            MKLGemm<T> functor(queue);
#endif

            status |=
                functor(transa, transb, m, n, k, (T)alpha, a_buffer_t, lda, offsetA, b_buffer_t, ldb, offsetB, (T)beta, c_buffer_t, ldc, offsetC);
        }
    };

public:
    static void run(::sycl::queue & queue, const math::Transpose transa, const math::Transpose transb, const size_t m, const size_t n, const size_t k,
                    const double alpha, const UniversalBuffer & a_buffer, const size_t lda, const size_t offsetA, const UniversalBuffer & b_buffer,
                    const size_t ldb, const size_t offsetB, const double beta, UniversalBuffer & c_buffer, const size_t ldc, const size_t offsetC,
                    Status & status)
    {
        DAAL_ASSERT(!a_buffer.empty());
        DAAL_ASSERT(!b_buffer.empty());
        DAAL_ASSERT(!c_buffer.empty());
        DAAL_ASSERT(a_buffer.type() == b_buffer.type());
        DAAL_ASSERT(b_buffer.type() == c_buffer.type());

        Execute op(queue, transa, transb, m, n, k, alpha, a_buffer, lda, offsetA, b_buffer, ldb, offsetB, beta, c_buffer, ldc, offsetC);
        TypeDispatcher::floatDispatch(a_buffer.type(), op, status);
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
        ::sycl::queue & queue;
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

        explicit Execute(::sycl::queue & queue, const math::UpLo upper_lower, const math::Transpose trans, const size_t n, const size_t k,
                         const double alpha, const UniversalBuffer & a_buffer, const size_t lda, const size_t offsetA, const double beta,
                         UniversalBuffer & c_buffer, const size_t ldc, const size_t offsetC)
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
              offsetC(offsetC)
        {}

        size_t getAExpectedSize() const { return (trans == math::Transpose::NoTrans) ? lda * k : lda * n; }

        size_t getCExpectedSize() const { return (trans == math::Transpose::NoTrans) ? ldc * n : ldc * k; }

        template <typename T>
        void operator()(Typelist<T>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER(a_buffer, T, getAExpectedSize() + offsetA);
            DAAL_ASSERT_UNIVERSAL_BUFFER(c_buffer, T, getCExpectedSize() + offsetC);

            auto a_buffer_t = a_buffer.template get<T>();
            auto c_buffer_t = c_buffer.template get<T>();

            const math::Transpose transInv = trans == math::Transpose::NoTrans ? math::Transpose::Trans : math::Transpose::NoTrans;

#ifdef ONEAPI_DAAL_NO_MKL_GPU_FUNC
            ReferenceGemm<T> functor;
            status |=
                functor(transInv, trans, k, k, n, (T)alpha, a_buffer_t, lda, offsetA, a_buffer_t, lda, offsetA, (T)beta, c_buffer_t, ldc, offsetC);
#else
            MKLSyrk<T> functor(queue);
            status |= functor(upper_lower, transInv, k, n, (T)alpha, a_buffer_t, lda, offsetA, (T)beta, c_buffer_t, ldc, offsetC);
#endif
        }
    };

public:
    static void run(::sycl::queue & queue, const math::UpLo upper_lower, const math::Transpose trans, const size_t n, const size_t k,
                    const double alpha, const UniversalBuffer & a_buffer, const size_t lda, const size_t offsetA, const double beta,
                    UniversalBuffer & c_buffer, const size_t ldc, const size_t offsetC, Status & status)
    {
        DAAL_ASSERT(!a_buffer.empty());
        DAAL_ASSERT(!c_buffer.empty());
        DAAL_ASSERT(a_buffer.type() == c_buffer.type());

        Execute op(queue, upper_lower, trans, n, k, alpha, a_buffer, lda, offsetA, beta, c_buffer, ldc, offsetC);
        TypeDispatcher::floatDispatch(a_buffer.type(), op, status);
    }
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__AXPYEXECUTOR"></a>
 *  \brief Adapter for AXPY routine
 */
class AxpyExecutor
{
private:
    template <typename algorithmFPType>
    static Status checkSize(const int n, const Buffer<algorithmFPType> & buffer, const int inc)
    {
        return Status();
    }

    struct Execute
    {
        ::sycl::queue & queue;
        const uint32_t n;
        const double a;
        const UniversalBuffer & x_buffer;
        const int incx;
        UniversalBuffer & y_buffer;
        const int incy;

        explicit Execute(::sycl::queue & queue, const uint32_t n, const double a, const UniversalBuffer & x_buffer, const int incx,
                         UniversalBuffer & y_buffer, const int incy)
            : queue(queue), n(n), a(a), x_buffer(x_buffer), incx(incx), y_buffer(y_buffer), incy(incy)
        {}

        template <typename algorithmFPType>
        void operator()(Typelist<algorithmFPType>, Status & status)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(x_buffer, algorithmFPType);
            DAAL_ASSERT_UNIVERSAL_BUFFER_TYPE(y_buffer, algorithmFPType);

            auto x_buffer_t = x_buffer.template get<algorithmFPType>();
            auto y_buffer_t = y_buffer.template get<algorithmFPType>();

            DAAL_ASSERT(n > 0);
            DAAL_ASSERT(size_t((n - 1) * incx) < x_buffer_t.size());
            DAAL_ASSERT(size_t((n - 1) * incy) < y_buffer_t.size());

#ifdef ONEAPI_DAAL_NO_MKL_GPU_FUNC
            ReferenceAxpy<algorithmFPType> functor;
#else
            MKLAxpy<algorithmFPType> functor(queue);
#endif
            status |= functor(n, (algorithmFPType)a, x_buffer_t, incx, y_buffer_t, incy);
        }
    };

public:
    static void run(::sycl::queue & queue, const uint32_t n, const double a, const UniversalBuffer x_buffer, const int incx, UniversalBuffer y_buffer,
                    const int incy, Status & status)
    {
        DAAL_ASSERT(!x_buffer.empty());
        DAAL_ASSERT(!y_buffer.empty());
        DAAL_ASSERT(x_buffer.type() == y_buffer.type());

        Execute op(queue, n, a, x_buffer, incx, y_buffer, incy);
        TypeDispatcher::floatDispatch(x_buffer.type(), op, status);
    }
};

/** @} */

} // namespace interface1

using interface1::GemmExecutor;
using interface1::SyrkExecutor;
using interface1::AxpyExecutor;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
