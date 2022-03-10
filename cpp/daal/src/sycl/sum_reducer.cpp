/* file: sum_reducer.cpp */
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

#include "src/sycl/reducer.h"
#include "services/internal/execution_context.h"
#include "src/externals/service_profiler.h"
#include "services/daal_defines.h"

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
services::Status buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId)
{
    services::Status status;

    services::String fptype_name = getKeyFPType(vectorTypeId);
    auto build_options           = fptype_name;
    build_options.add("-cl-std=CL1.2 -D LOCAL_BUFFER_SIZE=256");

    services::String cachekey("__daal_oneapi_internal_math_sum_reducer_");
    cachekey.add(build_options);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), sum_reducer, build_options.c_str(), status);

    return status;
}

services::Status sum_singlepass(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & vectors,
                                uint32_t nVectors, uint32_t vectorSize, SumReducer::Result & result)
{
    services::Status status;

    auto sum_kernel = kernelFactory.getKernel("sum_singlesubgroup", status);
    DAAL_CHECK_STATUS_VAR(status);

    // no need to check overflow for nVectors * vectorSize due to we already have buffer vectors of such size
    if (vectors.type() == TypeIds::id<float>())
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(vectors, float, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.sum, float, nVectors);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.sumOfSquares, float, nVectors);
    }
    else
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(vectors, double, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.sum, double, nVectors);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.sumOfSquares, double, nVectors);
    }

    const uint32_t maxWorkItemsPerSubGroup = 32;

    KernelRange localRange(1, maxWorkItemsPerSubGroup);
    KernelRange globalRange(nVectors, maxWorkItemsPerSubGroup);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(5, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, vectors, AccessModeIds::read);
    args.set(1, nVectors);
    args.set(2, vectorSize);
    args.set(3, result.sum, AccessModeIds::write);
    args.set(4, result.sumOfSquares, AccessModeIds::write);

    context.run(range, sum_kernel, args, status);
    return status;
}

services::Status runStepColmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & vectors,
                                 uint32_t nVectors, uint32_t vectorSize, uint32_t numWorkItems, uint32_t numWorkGroups, uint32_t numDivisionsByRow,
                                 SumReducer::Result & stepResult)
{
    services::Status status;

    auto sum_kernel = kernelFactory.getKernel("sum_step_colmajor", status);
    DAAL_CHECK_STATUS_VAR(status);

    // no need to check overflow for nVectors * vectorSize due to we already have buffer vectors of such size
    if (vectors.type() == TypeIds::id<float>())
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(vectors, float, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.sum, float, nVectors * numDivisionsByRow);
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.sumOfSquares, float, nVectors * numDivisionsByRow);
    }
    else
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(vectors, double, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.sum, double, nVectors * numDivisionsByRow);
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.sumOfSquares, double, nVectors * numDivisionsByRow);
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, numWorkGroups, numWorkItems);

    KernelRange localRange(numWorkItems);
    KernelRange globalRange(numWorkGroups * numWorkItems);

    KernelNDRange range(1);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(5, status);
    DAAL_CHECK_STATUS_VAR(status);

    args.set(0, vectors, AccessModeIds::read);
    args.set(1, nVectors);
    args.set(2, vectorSize);
    args.set(3, stepResult.sum, AccessModeIds::write);
    args.set(4, stepResult.sumOfSquares, AccessModeIds::write);

    context.run(range, sum_kernel, args, status);

    return status;
}

services::Status runFinalStepRowmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, SumReducer::Result & stepResult,
                                      uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup, SumReducer::Result & result)
{
    services::Status status;

    auto sum_kernel = kernelFactory.getKernel("sum_final_step_rowmajor", status);
    DAAL_CHECK_STATUS_VAR(status);

    if (result.sum.type() == TypeIds::id<float>())
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.sum, float, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.sumOfSquares, float, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.sum, float, nVectors);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.sumOfSquares, float, nVectors);
    }
    else
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.sum, double, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.sumOfSquares, double, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.sum, double, nVectors);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.sumOfSquares, double, nVectors);
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, workItemsPerGroup, nVectors);

    KernelRange localRange(workItemsPerGroup);
    KernelRange globalRange(workItemsPerGroup * nVectors);

    KernelNDRange range(1);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(6, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, stepResult.sum, AccessModeIds::read);
    args.set(1, stepResult.sumOfSquares, AccessModeIds::read);
    args.set(2, nVectors);
    args.set(3, vectorSize);
    args.set(4, result.sum, AccessModeIds::write);
    args.set(5, result.sumOfSquares, AccessModeIds::write);

    context.run(range, sum_kernel, args, status);

    return status;
}
SumReducer::Result SumReducer::sum(Layout vectorsLayout, const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize,
                                   services::Status & status)
{
    auto & context = services::internal::getDefaultContext();
    Result result(context, nVectors, vectors.type(), status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);
    return sum(vectorsLayout, vectors, nVectors, vectorSize, result, status);
}
SumReducer::Result SumReducer::sum(Layout vectorsLayout, const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize, Result & result,
                                   services::Status & status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(SumReducer.sum);

    auto & context       = services::internal::getDefaultContext();
    auto & kernelFactory = context.getClKernelFactory();

    status |= buildProgram(kernelFactory, vectors.type());
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, SumReducer::Result());

    DAAL_ASSERT(vectors.type() == TypeIds::id<float>() || vectors.type() == TypeIds::id<double>());

    const uint32_t maxWorkItemsPerGroup = 256;
    const uint32_t maxNumSubSlices      = 9;

    if (vectorsLayout == Layout::RowMajor)
    {
        status |= sum_singlepass(context, kernelFactory, vectors, nVectors, vectorSize, result);
    }
    else
    {
        const uint32_t numDivisionsByCol = (nVectors + maxWorkItemsPerGroup - 1) / maxWorkItemsPerGroup;
        uint32_t numDivisionsByRow       = 9;
        if (vectorSize < 5000)
            numDivisionsByRow = 1;
        else if (vectorSize < 10000)
            numDivisionsByRow = 3;
        else if (vectorSize < 20000)
            numDivisionsByRow = 6;

        const uint32_t workItemsPerGroup = (maxWorkItemsPerGroup < nVectors) ? maxWorkItemsPerGroup : nVectors;

        if (numDivisionsByRow > 1)
        {
            // no need to check overflow for numDivisionsByRow * nVectors due to numDivisionsByRow less than vectorSize,
            // and input vectors buffer has size of vectorSize * numDivisionsByRow
            Result stepResult(context, numDivisionsByRow * nVectors, vectors.type(), status);
            DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);

            status |= runStepColmajor(context, kernelFactory, vectors, nVectors, vectorSize, workItemsPerGroup, numDivisionsByCol * numDivisionsByRow,
                                      numDivisionsByRow, stepResult);
            DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);

            const uint32_t stepWorkItems = maxNumSubSlices / 2; //need to be power of two
            status |= runFinalStepRowmajor(context, kernelFactory, stepResult, nVectors, numDivisionsByRow, stepWorkItems, result);
        }
        else
        {
            status |= runStepColmajor(context, kernelFactory, vectors, nVectors, vectorSize, workItemsPerGroup, numDivisionsByCol, numDivisionsByRow,
                                      result);
        }
    }

    return result;
}

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
