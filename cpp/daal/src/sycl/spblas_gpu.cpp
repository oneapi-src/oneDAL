/* file: spblas_gpu.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "src/sycl/spblas_gpu.h"
#include "src/sycl/cl_kernels/kernel_sparse_blas.cl"

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
services::Status SpBlasGpu<algorithmFPType>::xgemm(const Transpose transa, const Transpose transb, const size_t m, const size_t n, const size_t k,
                                                   const algorithmFPType alpha, const services::internal::Buffer<algorithmFPType> & a_buffer,
                                                   const services::internal::Buffer<size_t> & aColsBuff,
                                                   const services::internal::Buffer<size_t> & aRowIndBuff,
                                                   const services::internal::Buffer<algorithmFPType> & b_buffer,
                                                   const services::internal::Buffer<size_t> & bColsBuff,
                                                   const services::internal::Buffer<size_t> & bRowIndBuff, const algorithmFPType beta,
                                                   services::internal::Buffer<algorithmFPType> & c_buffer, const size_t ldc, const size_t offsetC)
{
    services::Status status;

    auto & context            = services::internal::getDefaultContext();
    auto & factory            = context.getClKernelFactory();
    services::String options  = getKeyFPType<algorithmFPType>();
    services::String cacheKey = "__daal_services_math_spmm_";
    cacheKey.add(options);

    factory.build(ExecutionTargetIds::device, cacheKey.c_str(), clKernelSpGemm, options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);
    const char * const kernelName = beta != algorithmFPType(0) ? "spmm_kernel" : "spmm_kernel_without_sum";
    KernelPtr kernelSpGemm        = factory.getKernel(kernelName, status);
    DAAL_CHECK_STATUS_VAR(status);

    const size_t one = size_t(1);

    if (transa == Transpose::Trans && transb == Transpose::NoTrans)
    {
        KernelArguments args(11, status);
        DAAL_CHECK_STATUS_VAR(status);

        args.set(0, alpha);
        args.set(1, a_buffer);
        args.set(2, aColsBuff);
        args.set(3, aRowIndBuff);
        args.set(4, b_buffer);
        args.set(5, bColsBuff);
        args.set(6, bRowIndBuff);
        args.set(7, c_buffer);
        args.set(8, ldc);
        args.set(9, offsetC);
        args.set(10, beta);
        KernelRange range(m, n);
        context.run(range, kernelSpGemm, args, status);
    }
    else
    {
        return services::ErrorMethodNotImplemented;
    }

    return status;
}

template class SpBlasGpu<float>;
template class SpBlasGpu<double>;

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
