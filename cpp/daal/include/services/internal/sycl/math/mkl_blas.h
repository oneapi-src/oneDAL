/* file: mkl_blas.h */
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

/*
//++
//  Wrappers for BLAS functions.
//--
*/

#ifndef __ONEAPI_INTERNAL_MKL_BLAS_H__
#define __ONEAPI_INTERNAL_MKL_BLAS_H__

#include "services/internal/buffer.h"
#include "services/internal/sycl/error_handling_sycl.h"
#include "services/internal/sycl/math/mkl_dal_utils.h"

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
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLGEMM"></a>
 *  \brief Adapter for Intel(R) MKL GEMM routine
 */
template <typename algorithmFPType>
struct MKLGemm
{
    MKLGemm(::sycl::queue & queue) : _queue(queue) {}

    Status operator()(const math::Transpose transa, const math::Transpose transb, const size_t m, const size_t n, const size_t k,
                      const algorithmFPType alpha, const Buffer<algorithmFPType> & a_buffer, const size_t lda, const size_t offsetA,
                      const Buffer<algorithmFPType> & b_buffer, const size_t ldb, const size_t offsetB, const algorithmFPType beta,
                      Buffer<algorithmFPType> & c_buffer, const size_t ldc, const size_t offsetC)
    {
        Status status;

#ifdef DAAL_SYCL_INTERFACE_USM
        const auto transamkl = to_fpk_transpose(transa);
        const auto transbmkl = to_fpk_transpose(transb);

        auto a_usm = a_buffer.toUSM(_queue, data_management::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        auto b_usm = b_buffer.toUSM(_queue, data_management::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        auto c_usm = c_buffer.toUSM(_queue, data_management::readWrite, status);
        DAAL_CHECK_STATUS_VAR(status);

        auto a_ptr = a_usm.get() + offsetA;
        auto b_ptr = b_usm.get() + offsetB;
        auto c_ptr = c_usm.get() + offsetC;

        status |= catchSyclExceptions([&]() mutable {
            ::oneapi::fpk::blas::gemm(_queue, transamkl, transbmkl, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
            _queue.wait_and_throw();
        });
#else
        static_assert(false, "USM support required");
#endif
        return status;
    }

private:
    template <typename T>
    void innerGemm(MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, T alpha, ::sycl::buffer<T, 1> a, int64_t lda,
                   ::sycl::buffer<T, 1> b, int64_t ldb, T beta, ::sycl::buffer<T, 1> c, int64_t ldc, int64_t offset_a, int64_t offset_b,
                   int64_t offset_c);

    template <>
    void innerGemm<double>(MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, double alpha, ::sycl::buffer<double, 1> a,
                           int64_t lda, ::sycl::buffer<double, 1> b, int64_t ldb, double beta, ::sycl::buffer<double, 1> c, int64_t ldc,
                           int64_t offset_a, int64_t offset_b, int64_t offset_c)
    {
        ::oneapi::fpk::gpu::dgemm_sycl(&_queue, transa, transb, m, n, k, alpha, &a, lda, &b, ldb, beta, &c, ldc, offset_a, offset_b, offset_c);
    }

    template <>
    void innerGemm<float>(MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha, ::sycl::buffer<float, 1> a,
                          int64_t lda, ::sycl::buffer<float, 1> b, int64_t ldb, float beta, ::sycl::buffer<float, 1> c, int64_t ldc, int64_t offset_a,
                          int64_t offset_b, int64_t offset_c)
    {
        ::oneapi::fpk::gpu::sgemm_sycl(&_queue, transa, transb, m, n, k, alpha, &a, lda, &b, ldb, beta, &c, ldc, offset_a, offset_b, offset_c);
    }

    ::sycl::queue & _queue;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLSYRK"></a>
 *  \brief Adapter for Intel(R) MKL SYRK routine
 */
template <typename algorithmFPType>
struct MKLSyrk
{
    MKLSyrk(::sycl::queue & queue) : _queue(queue) {}

    Status operator()(const math::UpLo upper_lower, const math::Transpose trans, const size_t n, const size_t k, const algorithmFPType alpha,
                      const Buffer<algorithmFPType> & a_buffer, const size_t lda, const size_t offsetA, const algorithmFPType beta,
                      Buffer<algorithmFPType> & c_buffer, const size_t ldc, const size_t offsetC)
    {
        Status status;

#ifdef DAAL_SYCL_INTERFACE_USM
        const auto transmkl = to_fpk_transpose(trans);
        const auto uplomkl  = to_fpk_uplo(upper_lower);

        auto a_usm = a_buffer.toUSM(_queue, data_management::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        auto c_usm = c_buffer.toUSM(_queue, data_management::readWrite, status);
        DAAL_CHECK_STATUS_VAR(status);

        auto a_ptr = a_usm.get() + offsetA;
        auto c_ptr = c_usm.get() + offsetC;

        status |= catchSyclExceptions([&]() mutable {
            ::oneapi::fpk::blas::syrk(_queue, uplomkl, transmkl, n, k, alpha, a_ptr, lda, beta, c_ptr, ldc);
            _queue.wait_and_throw();
        });
#else
        static_assert(false, "USM support required");
#endif
        return status;
    }

private:
    template <typename T>
    void innerSyrk(MKL_UPLO uplo, MKL_TRANSPOSE trans, int64_t n, int64_t k, T alpha, ::sycl::buffer<T, 1> a, int64_t lda, T beta,
                   ::sycl::buffer<T, 1> c, int64_t ldc, int64_t offset_a, int64_t offset_c);

    template <>
    void innerSyrk(MKL_UPLO uplo, MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha, ::sycl::buffer<double, 1> a, int64_t lda, double beta,
                   ::sycl::buffer<double, 1> c, int64_t ldc, int64_t offset_a, int64_t offset_c)
    {
        ::oneapi::fpk::gpu::dsyrk_sycl(&_queue, uplo, trans, n, k, alpha, &a, lda, beta, &c, ldc, offset_a, offset_c);
    }

    template <>
    void innerSyrk(MKL_UPLO uplo, MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha, ::sycl::buffer<float, 1> a, int64_t lda, float beta,
                   ::sycl::buffer<float, 1> c, int64_t ldc, int64_t offset_a, int64_t offset_c)
    {
        ::oneapi::fpk::gpu::ssyrk_sycl(&_queue, uplo, trans, n, k, alpha, &a, lda, beta, &c, ldc, offset_a, offset_c);
    }

    ::sycl::queue & _queue;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLAXPY"></a>
 *  \brief Adapter for Intel(R) MKL AXPY routine
 */
template <typename algorithmFPType>
struct MKLAxpy
{
    MKLAxpy(::sycl::queue & queue) : _queue(queue) {}

    Status operator()(const int n, const algorithmFPType a, const Buffer<algorithmFPType> & x_buffer, const int incx,
                      Buffer<algorithmFPType> & y_buffer, const int incy)
    {
        Status status;

#ifdef DAAL_SYCL_INTERFACE_USM
        auto x_usm = x_buffer.toUSM(_queue, data_management::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);

        auto y_usm = y_buffer.toUSM(_queue, data_management::readWrite, status);
        DAAL_CHECK_STATUS_VAR(status);

        status |= catchSyclExceptions([&]() mutable {
            ::oneapi::fpk::blas::axpy(_queue, n, a, x_usm.get(), incx, y_usm.get(), incy);
            _queue.wait_and_throw();
        });
#else
        static_assert(false, "USM support required");
#endif
        return status;
    }

private:
    ::sycl::queue & _queue;
};

/** @} */
} // namespace interface1

using interface1::MKLGemm;
using interface1::MKLSyrk;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
