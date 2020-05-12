/* file: reducer.cpp */
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

#include "service/kernel/oneapi/reducer.h"
#include "services/env_detect.h"
#include "externals/service_ittnotify.h"
#include "service/kernel/oneapi/cl_kernels/op_reducer.cl"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{
DAAL_ITTNOTIFY_DOMAIN(daal.oneapi.internal.math.Reducer);

services::Status Reducer::buildProgram(ClKernelFactoryIface & kernelFactory, const BinaryOp op, const TypeId & vectorTypeId)
{
    services::String fptype_name = getKeyFPType(vectorTypeId);
    auto build_options           = fptype_name;
    build_options.add(" -cl-std=CL1.2 -D LOCAL_BUFFER_SIZE=256");

    if (op == BinaryOp::MIN)
    {
        build_options.add(" -D UNARY_OP=none -D BINARY_OP=min -D INIT_VALUE=FLT_MAX");
    }
    else if (op == BinaryOp::MAX)
    {
        build_options.add(" -D UNARY_OP=none -D BINARY_OP=max -D INIT_VALUE=-FLT_MAX");
    }
    else if (op == BinaryOp::SUM)
    {
        build_options.add(" -D UNARY_OP=none -D BINARY_OP=sum -D INIT_VALUE=0.0");
    }
    else if (op == BinaryOp::SUM_OF_SQUARES)
    {
        build_options.add(" -D UNARY_OP=pow2 -D BINARY_OP=sum -D INIT_VALUE=0.0");
    }

    services::String cachekey("__daal_oneapi_internal_math_reducer_");
    cachekey.add(build_options);

    services::Status status;
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), op_reduce, build_options.c_str(), &status);
    return status;
}

void Reducer::singlepass(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, Layout vectorsLayout, const UniversalBuffer & vectors,
                         uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup, UniversalBuffer & reduceRes, services::Status * status)
{
    auto reduce_kernel = kernelFactory.getKernel("reduceSinglepass");

    KernelRange localRange(workItemsPerGroup, 1);
    KernelRange globalRange(workItemsPerGroup, nVectors);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(5);
    uint32_t vectorsAreRows = vectorsLayout == Layout::RowMajor ? 1 : 0;
    args.set(0, vectorsAreRows);
    args.set(1, vectors, AccessModeIds::read);
    args.set(2, nVectors);
    args.set(3, vectorSize);
    args.set(4, reduceRes, AccessModeIds::write);

    context.run(range, reduce_kernel, args, status);
}

void Reducer::runStepColmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & vectors,
                              uint32_t nVectors, uint32_t vectorSize, uint32_t numWorkItems, uint32_t numWorkGroups, Reducer::Result & stepResult,
                              services::Status * status)
{
    auto reduce_kernel = kernelFactory.getKernel("reduceStepColmajor");

    KernelRange localRange(numWorkItems);
    KernelRange globalRange(numWorkGroups * numWorkItems);

    KernelNDRange range(1);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(4);

    args.set(0, vectors, AccessModeIds::read);
    args.set(1, nVectors);
    args.set(2, vectorSize);
    args.set(3, stepResult.reduceRes, AccessModeIds::write);

    context.run(range, reduce_kernel, args, status);
}

void Reducer::runFinalStepRowmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, Reducer::Result & stepResult,
                                   uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup, Reducer::Result & result,
                                   services::Status * status)
{
    auto reduce_kernel = kernelFactory.getKernel("reduceFinalStepRowmajor");

    KernelRange localRange(workItemsPerGroup);
    KernelRange globalRange(workItemsPerGroup * nVectors);

    KernelNDRange range(1);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_PTR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_PTR(status);

    KernelArguments args(4);
    args.set(0, stepResult.reduceRes, AccessModeIds::read);
    args.set(1, nVectors);
    args.set(2, vectorSize);
    args.set(3, result.reduceRes, AccessModeIds::write);

    context.run(range, reduce_kernel, args, status);
}

Reducer::Result Reducer::reduce(const BinaryOp op, Layout vectorsLayout, const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize,
                                services::Status * status)
{
    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    Result result(context, nVectors, vectors.type(), status);
    return Reducer::reduce(op, vectorsLayout, vectors, result.reduceRes, nVectors, vectorSize, status);
}

Reducer::Result Reducer::reduce(const BinaryOp op, Layout vectorsLayout, const UniversalBuffer & vectors, UniversalBuffer & resReduce,
                                uint32_t nVectors, uint32_t vectorSize, services::Status * status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(Reducer);
    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    Result result(context, resReduce, nVectors, vectors.type(), status);

    auto & kernelFactory = context.getClKernelFactory();

    buildProgram(kernelFactory, op, vectors.type());

    const uint32_t maxWorkItemsPerGroup = 256;
    const uint32_t maxNumSubSlices      = 9;

    if (vectorsLayout == Layout::RowMajor)
    {
        singlepass(context, kernelFactory, vectorsLayout, vectors, nVectors, vectorSize, maxWorkItemsPerGroup, resReduce, status);
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
            DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, stepResult);

            runStepColmajor(context, kernelFactory, vectors, nVectors, vectorSize, workItemsPerGroup, numDivisionsByCol * numDivisionsByRow,
                            stepResult, status);

            const uint32_t stepWorkItems = maxNumSubSlices / 2; //need to be power of two
            runFinalStepRowmajor(context, kernelFactory, stepResult, nVectors, numDivisionsByRow, stepWorkItems, result, status);
        }
        else
        {
            runStepColmajor(context, kernelFactory, vectors, nVectors, vectorSize, workItemsPerGroup, numDivisionsByCol, result, status);
        }
    }

    return result;
}

} // namespace math
} // namespace internal
} // namespace oneapi
} // namespace daal
