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

#include "src/sycl/reducer.h"
#include "services/internal/execution_context.h"
#include "src/externals/service_profiler.h"
#include "src/sycl/cl_kernels/op_reducer.cl"
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
    kernelFactory.build(ExecutionTargetIds::device, cachekey.c_str(), op_reduce, build_options.c_str(), status);
    return status;
}

services::Status Reducer::singlepass(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, Layout vectorsLayout,
                                     const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup,
                                     UniversalBuffer & reduceRes)
{
    services::Status status;
    auto reduce_kernel = kernelFactory.getKernel("reduceSinglepass", status);
    DAAL_CHECK_STATUS_VAR(status);

    // no need to check overflow for nVectors * vectorSize due to we already have buffer vectors of such size
    if (vectors.type() == TypeIds::id<float>())
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(vectors, float, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(reduceRes, float, nVectors);
    }
    else
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(vectors, double, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(reduceRes, double, nVectors);
    }

    KernelRange localRange(workItemsPerGroup, 1);
    KernelRange globalRange(workItemsPerGroup, nVectors);

    KernelNDRange range(2);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(5, status);
    DAAL_CHECK_STATUS_VAR(status);
    uint32_t vectorsAreRows = vectorsLayout == Layout::RowMajor ? 1 : 0;
    args.set(0, vectorsAreRows);
    args.set(1, vectors, AccessModeIds::read);
    args.set(2, nVectors);
    args.set(3, vectorSize);
    args.set(4, reduceRes, AccessModeIds::write);

    context.run(range, reduce_kernel, args, status);
    return status;
}

services::Status Reducer::runStepColmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & vectors,
                                          uint32_t nVectors, uint32_t vectorSize, uint32_t numWorkItems, uint32_t numWorkGroups,
                                          uint32_t numDivisionsByRow, Reducer::Result & stepResult)
{
    services::Status status;
    auto reduce_kernel = kernelFactory.getKernel("reduceStepColmajor", status);
    DAAL_CHECK_STATUS_VAR(status);

    // no need to check overflow for nVectors * vectorSize due to we already have buffer vectors of such size
    if (vectors.type() == TypeIds::id<float>())
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(vectors, float, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.reduceRes, float, nVectors * numDivisionsByRow);
    }
    else
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(vectors, double, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.reduceRes, double, nVectors * numDivisionsByRow);
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, numWorkGroups, numWorkItems);

    KernelRange localRange(numWorkItems);
    KernelRange globalRange(numWorkGroups * numWorkItems);

    KernelNDRange range(1);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(4, status);
    DAAL_CHECK_STATUS_VAR(status);

    args.set(0, vectors, AccessModeIds::read);
    args.set(1, nVectors);
    args.set(2, vectorSize);
    args.set(3, stepResult.reduceRes, AccessModeIds::write);

    context.run(range, reduce_kernel, args, status);
    return status;
}

services::Status Reducer::runFinalStepRowmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, Reducer::Result & stepResult,
                                               uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup, Reducer::Result & result)
{
    services::Status status;
    auto reduce_kernel = kernelFactory.getKernel("reduceFinalStepRowmajor", status);
    DAAL_CHECK_STATUS_VAR(status);

    // no need to check overflow for nVectors * vectorSize due to we already have buffer vectors of such size
    if (result.reduceRes.type() == TypeIds::id<float>())
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.reduceRes, float, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.reduceRes, float, nVectors);
    }
    else
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(stepResult.reduceRes, double, nVectors * vectorSize);
        DAAL_ASSERT_UNIVERSAL_BUFFER(result.reduceRes, double, nVectors);
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, workItemsPerGroup, nVectors);

    KernelRange localRange(workItemsPerGroup);
    KernelRange globalRange(workItemsPerGroup * nVectors);

    KernelNDRange range(1);
    range.global(globalRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    range.local(localRange, status);
    DAAL_CHECK_STATUS_VAR(status);

    KernelArguments args(4, status);
    DAAL_CHECK_STATUS_VAR(status);
    args.set(0, stepResult.reduceRes, AccessModeIds::read);
    args.set(1, nVectors);
    args.set(2, vectorSize);
    args.set(3, result.reduceRes, AccessModeIds::write);

    context.run(range, reduce_kernel, args, status);
    return status;
}

Reducer::Result Reducer::reduce(const BinaryOp op, Layout vectorsLayout, const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize,
                                services::Status & status)
{
    auto & context = services::internal::getDefaultContext();
    Result result(context, nVectors, vectors.type(), status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, Reducer::Result());
    return Reducer::reduce(op, vectorsLayout, vectors, result.reduceRes, nVectors, vectorSize, status);
}

Reducer::Result Reducer::reduce(const BinaryOp op, Layout vectorsLayout, const UniversalBuffer & vectors, UniversalBuffer & resReduce,
                                uint32_t nVectors, uint32_t vectorSize, services::Status & status)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(Reducer);
    auto & context = services::internal::getDefaultContext();

    Result result(context, resReduce, nVectors, vectors.type(), status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, Reducer::Result());

    DAAL_ASSERT(vectors.type() == TypeIds::id<float>() || vectors.type() == TypeIds::id<double>());

    auto & kernelFactory = context.getClKernelFactory();

    status |= buildProgram(kernelFactory, op, vectors.type());
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, result);

    const uint32_t maxWorkItemsPerGroup = 256;
    const uint32_t maxNumSubSlices      = 9;

    if (vectorsLayout == Layout::RowMajor)
    {
        status |= singlepass(context, kernelFactory, vectorsLayout, vectors, nVectors, vectorSize, maxWorkItemsPerGroup, resReduce);
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
