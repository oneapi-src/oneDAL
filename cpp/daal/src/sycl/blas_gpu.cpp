/* file: blas_gpu.cpp */
/*******************************************************************************
* Copyright 2015 Intel Corporation
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

#include "services/internal/sycl/math/reference_gemm.h"
#include "services/internal/sycl/math/reference_axpy.h"
#include "src/sycl/blas_gpu.h"
#include "src/sycl/cl_kernels/kernel_blas.cl"

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
template <typename algorithmFPType>
services::Status ReferenceGemm<algorithmFPType>::operator()(const Transpose transa, const Transpose transb, const size_t m, const size_t n,
                                                            const size_t k, const algorithmFPType alpha,
                                                            const services::internal::Buffer<algorithmFPType> & a_buffer, const size_t lda,
                                                            const size_t offsetA, const services::internal::Buffer<algorithmFPType> & b_buffer,
                                                            const size_t ldb, const size_t offsetB, const algorithmFPType beta,
                                                            services::internal::Buffer<algorithmFPType> & c_buffer, const size_t ldc,
                                                            const size_t offsetC)
{
    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();
    services::String options       = getKeyFPType<algorithmFPType>();
    services::String cacheKey      = "__daal_gemm_";
    cacheKey.add(options);

    factory.build(ExecutionTargetIds::device, cacheKey.c_str(), clKernelGemm, options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    const char * const kernelName = beta != algorithmFPType(0) ? "blas_sgemm_small" : "blas_sgemm_without_sum";

    KernelPtr kernelGemm = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(15, status);
    DAAL_CHECK_STATUS_VAR(status);

    const uint32_t one = uint32_t(1);

    args.set(0, (uint32_t)k);
    args.set(1, alpha);
    args.set(2, a_buffer);
    args.set(5, (uint32_t)offsetA);
    args.set(6, b_buffer);
    args.set(9, (uint32_t)offsetB);
    args.set(10, beta);
    args.set(11, c_buffer, AccessModeIds::write);
    args.set(12, one);
    args.set(13, (uint32_t)ldc);
    args.set(14, (uint32_t)offsetC);

    if (transa == Transpose::NoTrans && transb == Transpose::NoTrans)
    {
        args.set(3, one);
        args.set(4, (uint32_t)lda);
        args.set(7, one);
        args.set(8, (uint32_t)ldb);
    }
    else if (transa == Transpose::Trans && transb == Transpose::NoTrans)
    {
        args.set(3, (uint32_t)lda);
        args.set(4, one);
        args.set(7, one);
        args.set(8, (uint32_t)ldb);
    }
    else if (transa == Transpose::NoTrans && transb == Transpose::Trans)
    {
        args.set(3, one);
        args.set(4, (uint32_t)lda);
        args.set(7, (uint32_t)ldb);
        args.set(8, one);
    }
    else
    {
        args.set(3, one);
        args.set(4, (uint32_t)lda);
        args.set(7, one);
        args.set(8, (uint32_t)ldb);
    }

    KernelRange range(m, n);

    ctx.run(range, kernelGemm, args, status);

    return status;
}

template class ReferenceGemm<float>;
template class ReferenceGemm<double>;

template <typename algorithmFPType>
services::Status ReferenceAxpy<algorithmFPType>::operator()(const int n, const algorithmFPType a,
                                                            const services::internal::Buffer<algorithmFPType> & x_buffer, const int incx,
                                                            services::internal::Buffer<algorithmFPType> & y_buffer, const int incy)
{
    services::Status status;

    ExecutionContextIface & ctx    = services::internal::getDefaultContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();
    services::String options       = getKeyFPType<algorithmFPType>();
    services::String cacheKey      = "__daal_axpy_";
    cacheKey.add(options);

    factory.build(ExecutionTargetIds::device, cacheKey.c_str(), clKernelAxpy, options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelPtr blas_axpy = factory.getKernel("blas_axpy", status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(5, status);
    DAAL_CHECK_STATUS_VAR(status);

    args.set(0, a);
    args.set(1, x_buffer, AccessModeId::read);
    args.set(2, incx);
    args.set(3, y_buffer, AccessModeId::readwrite);
    args.set(4, incy);

    KernelRange range(n);

    ctx.run(range, blas_axpy, args, status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template class ReferenceAxpy<float>;
template class ReferenceAxpy<double>;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
