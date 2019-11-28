/* file: sum_reducer.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "oneapi/sum_reducer.h"
#include "services/env_detect.h"
#include "service_ittnotify.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{
DAAL_ITTNOTIFY_DOMAIN(daal.oneapi.internal.math.SumReducer);

void buildProgram(ClKernelFactoryIface & kernelFactory, const TypeId & vectorTypeId)
{
    services::String fptype_name = getKeyFPType(vectorTypeId);
    auto build_options           = fptype_name;
    build_options.add("-cl-std=CL1.2 -D LOCAL_BUFFER_SIZE=256");

    services::String cachekey("__daal_oneapi_internal_math_sum_reducer_");
    cachekey.add(fptype_name);
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), sum_reducer, build_options.c_str());
}

void sum_singlepass(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, Layout vectorsLayout, const UniversalBuffer & vectors,
                    uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup, SumReducer::Result & result, services::Status * status)
{
    auto sum_kernel = kernelFactory.getKernel("sum_singlepass");

    KernelRange localRange(workItemsPerGroup, 1);
    KernelRange globalRange(workItemsPerGroup, nVectors);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(6 /*8*/);
    uint32_t vectorsAreRows = vectorsLayout == Layout::RowMajor ? 1 : 0;
    args.set(0, vectorsAreRows);
    args.set(1, vectors, AccessModeIds::read);
    args.set(2, nVectors);
    args.set(3, vectorSize);
    args.set(4, result.sum, AccessModeIds::write);
    args.set(5, result.sumOfSquares, AccessModeIds::write);
    //args.set(6, LocalBuffer(vectors.type(), workItemsPerGroup));
    //args.set(7, LocalBuffer(vectors.type(), workItemsPerGroup));

    context.run(range, sum_kernel, args, status);
}

void run_step_colmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & vectors, uint32_t nVectors,
                       uint32_t vectorSize, uint32_t numWorkItems, uint32_t numWorkGroups, SumReducer::Result & stepResult, services::Status * status)
{
    auto sum_kernel = kernelFactory.getKernel("sum_step_colmajor");

    KernelRange localRange(numWorkItems);
    KernelRange globalRange(numWorkGroups * numWorkItems);

    KernelNDRange range(1);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(5);

    args.set(0, vectors, AccessModeIds::read);
    args.set(1, nVectors);
    args.set(2, vectorSize);
    args.set(3, stepResult.sum, AccessModeIds::write);
    args.set(4, stepResult.sumOfSquares, AccessModeIds::write);

    context.run(range, sum_kernel, args, status);
}

void run_final_step_rowmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, SumReducer::Result & stepResult,
                             uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup, SumReducer::Result & result,
                             services::Status * status)
{
    auto sum_kernel = kernelFactory.getKernel("sum_final_step_rowmajor");

    KernelRange localRange(workItemsPerGroup);
    KernelRange globalRange(workItemsPerGroup * nVectors);

    KernelNDRange range(1);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(6);
    args.set(0, stepResult.sum, AccessModeIds::read);
    args.set(1, stepResult.sumOfSquares, AccessModeIds::read);
    args.set(2, nVectors);
    args.set(3, vectorSize);
    args.set(4, result.sum, AccessModeIds::write);
    args.set(5, result.sumOfSquares, AccessModeIds::write);

    context.run(range, sum_kernel, args, status);
}

SumReducer::Result SumReducer::sum(Layout vectorsLayout, const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize,
                                   services::Status * status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(SumReducer.sum);

    auto & context       = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & kernelFactory = context.getClKernelFactory();

    buildProgram(kernelFactory, vectors.type());

    const uint32_t maxWorkItemsPerGroup = 256;
    const uint32_t maxNumSubSlices      = 9;

    Result result(context, nVectors, vectors.type(), status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);

    if (vectorsLayout == Layout::RowMajor)
    {
        sum_singlepass(context, kernelFactory, vectorsLayout, vectors, nVectors, vectorSize, maxWorkItemsPerGroup, result, status);
    }
    else
    {
        const int32_t numDivisionsByCol = (nVectors + maxWorkItemsPerGroup - 1) / maxWorkItemsPerGroup;
        int32_t numDivisionsByRow       = 9;
        if (vectorSize < 5000)
            numDivisionsByRow = 1;
        else if (vectorSize < 10000)
            numDivisionsByRow = 3;
        else if (vectorSize < 20000)
            numDivisionsByRow = 6;

        const int32_t workItemsPerGroup = (maxWorkItemsPerGroup < nVectors) ? maxWorkItemsPerGroup : nVectors;

        if (numDivisionsByRow > 1)
        {
            Result stepResult(context, numDivisionsByRow * nVectors, vectors.type(), status);
            DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);

            run_step_colmajor(context, kernelFactory, vectors, nVectors, vectorSize, workItemsPerGroup, numDivisionsByCol * numDivisionsByRow,
                              stepResult, status);

            const uint32_t stepWorkItems = maxNumSubSlices / 2; //need to be power of two
            run_final_step_rowmajor(context, kernelFactory, stepResult, nVectors, numDivisionsByRow, stepWorkItems, result, status);
        }
        else
        {
            run_step_colmajor(context, kernelFactory, vectors, nVectors, vectorSize, workItemsPerGroup, numDivisionsByCol, result, status);
        }
    }

    return result;
}

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal
