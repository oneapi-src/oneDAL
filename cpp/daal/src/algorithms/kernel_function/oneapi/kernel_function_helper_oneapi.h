/* file: kernel_function_helper_oneapi.h */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef __KERNEL_FUNCTION_HELPER_ONEAPI_H__
#define __KERNEL_FUNCTION_HELPER_ONEAPI_H__

#include "src/externals/service_profiler.h"
#include "src/externals/service_math.h"
#include "src/services/service_data_utils.h"
#include "src/algorithms/kernel_function/oneapi/cl_kernels/kernel_function.cl"
#include "src/sycl/math_service_types.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::services::internal::sycl;
using namespace daal::services::internal::sycl::math;

template <typename algorithmFPType>
class HelperKernel
{
public:
    static services::Status buildProgram(ClKernelFactoryIface & factory)
    {
        services::String options = getKeyFPType<algorithmFPType>();

        services::String cachekey("__daal_algorithms_kernel_function_rbf");
        cachekey.add(options);

        services::Status status;
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelKF, options.c_str(), status);
        return status;
    }

    static services::Status lazyAllocate(UniversalBuffer & x, const size_t n)
    {
        services::Status status;
        ExecutionContextIface & ctx = services::internal::getDefaultContext();
        const TypeIds::Id idType    = TypeIds::id<algorithmFPType>();
        if (x.empty() || x.get<algorithmFPType>().size() < n)
        {
            x = ctx.allocate(idType, n, status);
        }

        return status;
    }

    static services::Status sumOfSquaresCSR(const services::internal::Buffer<algorithmFPType> & valuesBuff,
                                            const services::internal::Buffer<size_t> & rowIndBuff, UniversalBuffer & sumOfSquaresBuff, const size_t n)

    {
        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("sumOfSquaresCSR", status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(rowIndBuff.size() == n + 1);
        DAAL_ASSERT_UNIVERSAL_BUFFER(sumOfSquaresBuff, algorithmFPType, n);

        args.set(0, valuesBuff, AccessModeIds::read);
        args.set(1, rowIndBuff, AccessModeIds::read);
        args.set(2, sumOfSquaresBuff, AccessModeIds::readwrite);

        KernelRange range(n);

        context.run(range, kernel, args, status);
        return status;
    }

    static services::Status computeRBF(const UniversalBuffer & sqrMatLeft, const UniversalBuffer & sqrMatRight, const uint32_t ld,
                                       const algorithmFPType coeff, services::internal::Buffer<algorithmFPType> & rbf, const size_t n, const size_t m)

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.computeRBF);

        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("computeRBF", status);
        DAAL_CHECK_STATUS_VAR(status);

        const algorithmFPType threshold = math::expThreshold<algorithmFPType>();

        KernelArguments args(6, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT_UNIVERSAL_BUFFER(sqrMatLeft, algorithmFPType, n);
        DAAL_ASSERT_UNIVERSAL_BUFFER(sqrMatRight, algorithmFPType, m);
        DAAL_ASSERT(rbf.size() == n * m);

        args.set(0, sqrMatLeft, AccessModeIds::read);
        args.set(1, sqrMatRight, AccessModeIds::read);
        args.set(2, ld);
        args.set(3, threshold);
        args.set(4, coeff);
        args.set(5, rbf, AccessModeIds::readwrite);

        KernelRange range(n, m);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }
};

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
