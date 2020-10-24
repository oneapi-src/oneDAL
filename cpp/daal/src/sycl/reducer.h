/* file: reducer.h */
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

#ifndef __REDUCER_H__
#define __REDUCER_H__

#include "src/sycl/math_service_types.h"
#include "services/internal/buffer.h"
#include "src/sycl/cl_kernels/sum_reducer.cl"
#include "services/internal/sycl/types_utils.h"
#include "services/internal/sycl/execution_context.h"

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
class SumReducer
{
public:
    SumReducer() = delete;

    struct Result
    {
        UniversalBuffer sum;
        UniversalBuffer sumOfSquares;

        Result() {}

        Result(ExecutionContextIface & context, uint32_t nVectors, TypeId type, services::Status & status)
            : sum(context.allocate(type, nVectors, status)), sumOfSquares(context.allocate(type, nVectors, status))
        {}
    };

public:
    static Result sum(Layout vectorsLayout, const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize, services::Status & status);
};

class Reducer
{
public:
    Reducer() = delete;

    enum class BinaryOp
    {
        MIN,
        MAX,
        SUM,
        SUM_OF_SQUARES
    };

    struct Result
    {
        UniversalBuffer reduceRes;

        Result() {}

        Result(ExecutionContextIface & context, uint32_t nVectors, TypeId type, services::Status & status)
            : reduceRes(context.allocate(type, nVectors, status))
        {}

        Result(ExecutionContextIface & context, UniversalBuffer & resReduce, uint32_t nVectors, TypeId type, services::Status & status)
            : reduceRes(resReduce)
        {}
    };

public:
    static Result reduce(const BinaryOp op, Layout vectorsLayout, const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize,
                         services::Status & status);
    static Result reduce(const BinaryOp op, Layout vectorsLayout, const UniversalBuffer & vectors, UniversalBuffer & resReduce, uint32_t nVectors,
                         uint32_t vectorSize, services::Status & status);

private:
    static services::Status buildProgram(ClKernelFactoryIface & kernelFactory, const BinaryOp op, const TypeId & vectorType);
    static services::Status singlepass(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, Layout vectorsLayout,
                                       const UniversalBuffer & vectors, uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup,
                                       UniversalBuffer & result);
    static services::Status runStepColmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, const UniversalBuffer & vectors,
                                            uint32_t nVectors, uint32_t vectorSize, uint32_t numWorkItems, uint32_t numWorkGroups,
                                            uint32_t numDivisionsByRow, Reducer::Result & stepResult);
    static services::Status runFinalStepRowmajor(ExecutionContextIface & context, ClKernelFactoryIface & kernelFactory, Reducer::Result & stepResult,
                                                 uint32_t nVectors, uint32_t vectorSize, uint32_t workItemsPerGroup, Reducer::Result & result);
};

} // namespace math
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

#endif
