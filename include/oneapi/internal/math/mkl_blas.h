/* file: mkl_blas.h */
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

/*
//++
//  Wrappers for BLAS functions.
//--
*/

#ifndef __ONEAPI_INTERNAL_MKL_BLAS_H__
#define __ONEAPI_INTERNAL_MKL_BLAS_H__

#include "services/buffer.h"
#include "mkl_dal_sycl.hpp"

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
/** @ingroup oneapi_internal
 * @{
 */

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLGEMM"></a>
 *  \brief Adapter for MKL GEMM routine
 */
template <typename algorithmFPType>
struct MKLGemm
{
    MKLGemm(cl::sycl::queue & queue) : _queue(queue) {}

    services::Status operator()(const math::Transpose transa, const math::Transpose transb, const size_t m, const size_t n, const size_t k,
                                const algorithmFPType alpha, const services::Buffer<algorithmFPType> & a_buffer, const size_t lda,
                                const size_t offsetA, const services::Buffer<algorithmFPType> & b_buffer, const size_t ldb, const size_t offsetB,
                                const algorithmFPType beta, services::Buffer<algorithmFPType> & c_buffer, const size_t ldc, const size_t offsetC)
    {
        services::Status status;

        const MKL_TRANSPOSE transamkl = transa == math::Transpose::Trans ? MKL_TRANS : MKL_NOTRANS;
        const MKL_TRANSPOSE transbmkl = transb == math::Transpose::Trans ? MKL_TRANS : MKL_NOTRANS;

        cl::sycl::buffer<algorithmFPType, 1> a_sycl_buff = a_buffer.toSycl();
        cl::sycl::buffer<algorithmFPType, 1> b_sycl_buff = b_buffer.toSycl();
        cl::sycl::buffer<algorithmFPType, 1> c_sycl_buff = c_buffer.toSycl();

        innerGemm(transamkl, transbmkl, m, n, k, alpha, a_sycl_buff, lda, b_sycl_buff, ldb, beta, c_sycl_buff, ldc, offsetA, offsetB, offsetC);

        _queue.wait();
        return status;
    }

private:
    template <typename T>
    void innerGemm(MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1> a, int64_t lda,
                   cl::sycl::buffer<T, 1> b, int64_t ldb, T beta, cl::sycl::buffer<T, 1> c, int64_t ldc, int64_t offset_a, int64_t offset_b,
                   int64_t offset_c);

    template <>
    void innerGemm<double>(MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, double alpha, cl::sycl::buffer<double, 1> a,
                           int64_t lda, cl::sycl::buffer<double, 1> b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> c, int64_t ldc,
                           int64_t offset_a, int64_t offset_b, int64_t offset_c)
    {
        fpk::gpu::dgemm_sycl(&_queue, transa, transb, m, n, k, alpha, &a, lda, &b, ldb, beta, &c, ldc, offset_a, offset_b, offset_c);
    }

    template <>
    void innerGemm<float>(MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> a,
                          int64_t lda, cl::sycl::buffer<float, 1> b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> c, int64_t ldc,
                          int64_t offset_a, int64_t offset_b, int64_t offset_c)
    {
        fpk::gpu::sgemm_sycl(&_queue, transa, transb, m, n, k, alpha, &a, lda, &b, ldb, beta, &c, ldc, offset_a, offset_b, offset_c);
    }

    cl::sycl::queue & _queue;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLSYRK"></a>
 *  \brief Adapter for MKL SYRK routine
 */
template <typename algorithmFPType>
struct MKLSyrk
{
    MKLSyrk(cl::sycl::queue & queue) : _queue(queue) {}

    services::Status operator()(const math::UpLo upper_lower, const math::Transpose trans, const size_t n, const size_t k,
                                const algorithmFPType alpha, const services::Buffer<algorithmFPType> & a_buffer, const size_t lda,
                                const size_t offsetA, const algorithmFPType beta, services::Buffer<algorithmFPType> & c_buffer, const size_t ldc,
                                const size_t offsetC)
    {
        services::Status status;

        const MKL_TRANSPOSE transmkl = trans == math::Transpose::Trans ? MKL_TRANS : MKL_NOTRANS;
        const MKL_UPLO uplomkl       = upper_lower == math::UpLo::Upper ? MKL_UPPER : MKL_LOWER;

        cl::sycl::buffer<algorithmFPType, 1> a_sycl_buff = a_buffer.toSycl();
        cl::sycl::buffer<algorithmFPType, 1> c_sycl_buff = c_buffer.toSycl();

        innerSyrk(uplomkl, transmkl, n, k, alpha, a_sycl_buff, lda, beta, c_sycl_buff, ldc, offsetA, offsetC);

        _queue.wait();
        return status;
    }

private:
    template <typename T>
    void innerSyrk(MKL_UPLO uplo, MKL_TRANSPOSE trans, int64_t n, int64_t k, T alpha, cl::sycl::buffer<T, 1> a, int64_t lda, T beta,
                   cl::sycl::buffer<T, 1> c, int64_t ldc, int64_t offset_a, int64_t offset_c);

    template <>
    void innerSyrk(MKL_UPLO uplo, MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha, cl::sycl::buffer<double, 1> a, int64_t lda, double beta,
                   cl::sycl::buffer<double, 1> c, int64_t ldc, int64_t offset_a, int64_t offset_c)
    {
        fpk::gpu::dsyrk_sycl(&_queue, uplo, trans, n, k, alpha, &a, lda, beta, &c, ldc, offset_a, offset_c);
    }

    template <>
    void innerSyrk(MKL_UPLO uplo, MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> a, int64_t lda, float beta,
                   cl::sycl::buffer<float, 1> c, int64_t ldc, int64_t offset_a, int64_t offset_c)
    {
        fpk::gpu::ssyrk_sycl(&_queue, uplo, trans, n, k, alpha, &a, lda, beta, &c, ldc, offset_a, offset_c);
    }

    cl::sycl::queue & _queue;
};

/**
 *  <a name="DAAL-CLASS-ONEAPI-INTERNAL__MKLAXPY"></a>
 *  \brief Adapter for MKL AXPY routine
 */
template <typename algorithmFPType>
struct MKLAxpy
{
    MKLAxpy(cl::sycl::queue & queue) : _queue(queue) {}

    services::Status operator()(const int n, const algorithmFPType a, const services::Buffer<algorithmFPType> & x_buffer, const int incx,
                                const services::Buffer<algorithmFPType> & y_buffer, const int incy)
    {
        cl::sycl::buffer<algorithmFPType, 1> x_sycl_buff = x_buffer.toSycl();
        cl::sycl::buffer<algorithmFPType, 1> y_sycl_buff = y_buffer.toSycl();

        innerAxpy(n, a, x_sycl_buff, incx, y_sycl_buff, incy);

        _queue.wait();
        return services::Status();
    }

private:
    template <typename T>
    void innerAxpy(const int n, const T a, cl::sycl::buffer<T, 1> & x_buffer, const int incx,
                   cl::sycl::buffer<T, 1> & y_buffer, const int incy);

    template <>
    void innerAxpy(const int n, const double a, cl::sycl::buffer<double, 1> & x_buffer, const int incx,
                   cl::sycl::buffer<double, 1> & y_buffer, const int incy)
    {
        fpk::blas::axpy(_queue, n, a, x_buffer, incx, y_buffer, incy);
    }

    template <>
    void innerAxpy(const int n, const float a, cl::sycl::buffer<float, 1> & x_buffer, const int incx,
                   cl::sycl::buffer<float, 1> & y_buffer, const int incy)
    {
        fpk::blas::axpy(_queue, n, a, x_buffer, incx, y_buffer, incy);
    }

    cl::sycl::queue & _queue;
};


/** @} */
} // namespace interface1

using interface1::MKLGemm;
using interface1::MKLSyrk;

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
