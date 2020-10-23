/* file: objective_function_utils_oneapi.h */
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

#ifndef __OBJECTIVE_FUNCTION_UTILS_H__
#define __OBJECTIVE_FUNCTION_UTILS_H__

#include "src/algorithms/objective_function/common/oneapi/cl_kernel/objective_function_utils.cl"
#include "src/data_management/service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace objective_function
{
namespace internal
{
using namespace daal::services::internal;

template <typename algorithmFPType>
struct HelperObjectiveFunction
{
    static services::Status lazyAllocate(services::internal::sycl::UniversalBuffer & x, const uint32_t n)
    {
        services::Status status;
        services::internal::sycl::ExecutionContextIface & ctx = services::internal::getDefaultContext();
        const services::internal::sycl::TypeIds::Id idType    = services::internal::sycl::TypeIds::id<algorithmFPType>();

        if (x.empty() || x.get<algorithmFPType>().size() < n)
        {
            x = ctx.allocate(idType, n, status);
        }

        return status;
    }

    static uint32_t getWorkgroupsCount(const uint32_t n, const uint32_t localWorkSize)
    {
        DAAL_ASSERT(localWorkSize > 0);
        const uint32_t elementsPerGroup = localWorkSize;
        uint32_t workgroupsCount        = n / elementsPerGroup;

        if (workgroupsCount * elementsPerGroup < n)
        {
            workgroupsCount++; // no need on overflow check since its always smaller than n
        }
        return workgroupsCount;
    }

    // sigma = (y - sigma)
    static services::Status subVectors(const services::internal::Buffer<algorithmFPType> & x, const services::internal::Buffer<algorithmFPType> & y,
                                       services::internal::Buffer<algorithmFPType> & result, const uint32_t n)
    {
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const char * const kernelName              = "subVectors";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(x.size() == n);
        DAAL_ASSERT(y.size() == n);
        DAAL_ASSERT(result.size() == n);

        args.set(0, x, services::internal::sycl::AccessModeIds::read);
        args.set(1, y, services::internal::sycl::AccessModeIds::read);
        args.set(2, result, services::internal::sycl::AccessModeIds::write);

        services::internal::sycl::KernelRange range(n);

        ctx.run(range, kernel, args, status);

        return status;
    }

    static services::Status setElem(const uint32_t index, const algorithmFPType element, services::internal::Buffer<algorithmFPType> & buffer)
    {
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const char * const kernelName              = "setElem";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(buffer.size() > index);

        services::internal::sycl::KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, index);
        args.set(1, element);
        args.set(2, buffer, services::internal::sycl::AccessModeIds::write);

        services::internal::sycl::KernelRange range(1);

        ctx.run(range, kernel, args, status);

        return status;
    }

    static services::Status setColElem(const uint32_t icol, const algorithmFPType element, services::internal::Buffer<algorithmFPType> & buffer,
                                       const uint32_t n, const uint32_t m)
    {
        services::Status status;
        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const char * const kernelName              = "setColElem";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK(icol < m, services::ErrorIncorrectParameter);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, n, m);
        DAAL_ASSERT(buffer.size() >= n * m);

        args.set(0, icol);
        args.set(1, element);
        args.set(2, buffer, services::internal::sycl::AccessModeIds::write);
        args.set(3, m);

        services::internal::sycl::KernelRange range(n);

        ctx.run(range, kernel, args, status);

        return status;
    }

    static services::Status transpose(const services::internal::Buffer<algorithmFPType> & x, services::internal::Buffer<algorithmFPType> & xt,
                                      const uint32_t n, const uint32_t p)
    {
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const char * const kernelName              = "transpose";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, n, p);
        DAAL_ASSERT(x.size() == n * p);
        DAAL_ASSERT(xt.size() == n * p);

        args.set(0, x, services::internal::sycl::AccessModeIds::read);
        args.set(1, xt, services::internal::sycl::AccessModeIds::write);
        args.set(2, n);
        args.set(3, p);

        services::internal::sycl::KernelRange range(n, p);

        ctx.run(range, kernel, args, status);

        return status;
    }

    static services::Status sumReduction(const services::internal::Buffer<algorithmFPType> & reductionBuffer, const size_t nWorkGroups,
                                         algorithmFPType & result)
    {
        services::Status status;

        auto sumReductionArrayPtr = reductionBuffer.toHost(data_management::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_ASSERT(reductionBuffer.size() == nWorkGroups);

        const auto * sumReductionArray = sumReductionArrayPtr.get();

        // Final summation with CPU
        for (size_t i = 0; i < nWorkGroups; i++)
        {
            result += sumReductionArray[i];
        }

        return status;
    }

    // l1*||beta|| + l2*||beta||**2
    static services::Status regularization(const services::internal::Buffer<algorithmFPType> & beta, const uint32_t nBeta, const uint32_t nClasses,
                                           algorithmFPType & reg, const algorithmFPType l1, const algorithmFPType l2)
    {
        services::Status status;
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nBeta, nClasses);
        const uint32_t n = nBeta * nClasses;

        const services::internal::sycl::TypeIds::Id idType = services::internal::sycl::TypeIds::id<algorithmFPType>();

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const char * const kernelName              = "regularization";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelNDRange range(1);

        size_t workItemsPerGroup = 256;

        const size_t nWorkGroups = getWorkgroupsCount(n, workItemsPerGroup);

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, workItemsPerGroup, nWorkGroups);
        services::internal::sycl::KernelRange localRange(workItemsPerGroup);
        services::internal::sycl::KernelRange globalRange(workItemsPerGroup * nWorkGroups);

        range.local(localRange, status);
        range.global(globalRange, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::UniversalBuffer buffer            = ctx.allocate(idType, nWorkGroups, status);
        DAAL_CHECK_STATUS_VAR(status);
        services::internal::Buffer<algorithmFPType> reductionBuffer = buffer.get<algorithmFPType>();

        services::internal::sycl::KernelArguments args(6, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(beta.size() == n);

        args.set(0, beta, services::internal::sycl::AccessModeIds::read);
        args.set(1, nBeta);
        args.set(2, n);
        args.set(3, reductionBuffer, services::internal::sycl::AccessModeIds::write);
        args.set(4, l1);
        args.set(5, l2);

        ctx.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS(status, sumReduction(reductionBuffer, nWorkGroups, reg));

        return status;
    }

    // s1 + s2 + .. + sn
    static services::Status sum(const services::internal::Buffer<algorithmFPType> & x, algorithmFPType & result, const uint32_t n)
    {
        services::Status status;
        const services::internal::sycl::TypeIds::Id idType = services::internal::sycl::TypeIds::id<algorithmFPType>();

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const char * const kernelName              = "sumReduction";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelNDRange range(1);

        size_t workItemsPerGroup = 256;

        const size_t nWorkGroups = getWorkgroupsCount(n, workItemsPerGroup);

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, workItemsPerGroup, nWorkGroups);
        services::internal::sycl::KernelRange localRange(workItemsPerGroup);
        services::internal::sycl::KernelRange globalRange(workItemsPerGroup * nWorkGroups);

        range.local(localRange, status);
        range.global(globalRange, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::UniversalBuffer buffer            = ctx.allocate(idType, nWorkGroups, status);
        DAAL_CHECK_STATUS_VAR(status);
        services::internal::Buffer<algorithmFPType> reductionBuffer = buffer.get<algorithmFPType>();

        services::internal::sycl::KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(x.size() == n);

        args.set(0, x, services::internal::sycl::AccessModeIds::read);
        args.set(1, n);
        args.set(2, reductionBuffer, services::internal::sycl::AccessModeIds::write);

        ctx.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS(status, sumReduction(reductionBuffer, nWorkGroups, result));

        return status;
    }

    // x = x + alpha
    // Where x - vector; alpha - scalar
    static services::Status addVectorScalar(services::internal::Buffer<algorithmFPType> & x, const algorithmFPType alpha, const uint32_t n)
    {
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const char * const kernelName              = "addVectorScalar";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(2, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_ASSERT(x.size() == n);

        args.set(0, x, services::internal::sycl::AccessModeIds::write);
        args.set(1, alpha);

        services::internal::sycl::KernelRange range(n);

        ctx.run(range, kernel, args, status);

        return status;
    }

    // x = x + y[id]
    // Where x - vector; y - vector, id - index
    static services::Status addVectorScalar(services::internal::Buffer<algorithmFPType> & x, const services::internal::Buffer<algorithmFPType> & y,
                                            const uint32_t id, const uint32_t n)
    {
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const char * const kernelName              = "addVectorScalar2";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(x.size() == n);
        DAAL_ASSERT(y.size() > id);

        args.set(0, x, services::internal::sycl::AccessModeIds::write);
        args.set(1, y, services::internal::sycl::AccessModeIds::read);
        args.set(2, id);

        services::internal::sycl::KernelRange range(n);

        ctx.run(range, kernel, args, status);

        return status;
    }

    static services::Status getXY(const services::internal::Buffer<algorithmFPType> & xBuff,
                                  const services::internal::Buffer<algorithmFPType> & yBuff, const services::internal::Buffer<int> & indBuff,
                                  services::internal::Buffer<algorithmFPType>& aX, services::internal::Buffer<algorithmFPType>& aY, uint32_t nBatch,
                                  uint32_t p, bool interceptFlag)
    {
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        status |= buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        const algorithmFPType interceptValue = interceptFlag ? algorithmFPType(1) : algorithmFPType(0);

        const char * const kernelName              = "getXY";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(7, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nBatch, (p+1));
        DAAL_ASSERT(xBuff.size() >= nBatch * p);
        DAAL_ASSERT(aX.size() == nBatch * (p + 1));
        DAAL_ASSERT(indBuff.size() == nBatch);
        // we do not check index values since they do not exceed n by the construction algorithm.

        DAAL_ASSERT(yBuff.size() >= nBatch);
        DAAL_ASSERT(aY.size() == nBatch);

        args.set(0, xBuff, services::internal::sycl::AccessModeIds::read);
        args.set(1, yBuff, services::internal::sycl::AccessModeIds::read);
        args.set(2, indBuff, services::internal::sycl::AccessModeIds::read);
        args.set(3, p);
        args.set(4, interceptValue);
        args.set(5, aX, services::internal::sycl::AccessModeIds::write);
        args.set(6, aY, services::internal::sycl::AccessModeIds::write);

        services::internal::sycl::KernelRange range(p, nBatch);

        ctx.run(range, kernel, args, status);

        return status;
    }

private:
    static services::Status buildProgram(services::internal::sycl::ClKernelFactoryIface & factory)
    {
        services::Status status;
        services::String options = services::internal::sycl::getKeyFPType<algorithmFPType>();

        services::String cachekey("__daal_algorithms_optimization_solver_objective_function_");
        cachekey.add(options);

        options.add(" -D LOCAL_SUM_SIZE=256 "); //depends on workItemsPerGroup value

        factory.build(services::internal::sycl::ExecutionTargetIds::device, cachekey.c_str(), clKernelObjectiveFunction, options.c_str(), status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }
};

} // namespace internal
} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
