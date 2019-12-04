/* file: objective_function_utils_oneapi.h */
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

#ifndef __OBJECTIVE_FUNCTION_UTILS_H__
#define __OBJECTIVE_FUNCTION_UTILS_H__

#include "cl_kernel/objective_function_utils.cl"
#include "service_numeric_table.h"

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
    static services::Status lazyAllocate(oneapi::internal::UniversalBuffer & x, const uint32_t n)
    {
        services::Status status;
        oneapi::internal::ExecutionContextIface & ctx = services::Environment::getInstance()->getDefaultExecutionContext();
        const oneapi::internal::TypeIds::Id idType    = oneapi::internal::TypeIds::id<algorithmFPType>();

        if (x.empty() || x.get<algorithmFPType>().size() < n)
        {
            x = ctx.allocate(idType, n, &status);
        }

        return status;
    }

    static uint32_t getWorkgroupsCount(const uint32_t n, const uint32_t localWorkSize)
    {
        const uint32_t elementsPerGroup = localWorkSize;
        uint32_t workgroupsCount        = n / elementsPerGroup;

        if (workgroupsCount * elementsPerGroup < n)
        {
            workgroupsCount++;
        }
        return workgroupsCount;
    }

    // sigma = (y - sigma)
    static services::Status subVectors(const services::Buffer<algorithmFPType> & x, const services::Buffer<algorithmFPType> & y,
                                       services::Buffer<algorithmFPType> & result, const uint32_t n)
    {
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "subVectors";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(3);
        args.set(0, x, oneapi::internal::AccessModeIds::read);
        args.set(1, y, oneapi::internal::AccessModeIds::read);
        args.set(2, result, oneapi::internal::AccessModeIds::write);

        oneapi::internal::KernelRange range(n);

        ctx.run(range, kernel, args, &status);

        return status;
    }

    static services::Status setElem(const uint32_t index, const algorithmFPType element, services::Buffer<algorithmFPType> & buffer)
    {
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "setElem";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(3);
        args.set(0, index);
        args.set(1, element);
        args.set(2, buffer, oneapi::internal::AccessModeIds::write);

        oneapi::internal::KernelRange range(1);

        ctx.run(range, kernel, args, &status);

        return status;
    }

    static services::Status setColElem(const uint32_t icol, const algorithmFPType element, services::Buffer<algorithmFPType> & buffer,
                                       const uint32_t n, const uint32_t m)
    {
        services::Status status;
        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "setColElem";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(4);
        args.set(0, icol);
        args.set(1, element);
        args.set(2, buffer, oneapi::internal::AccessModeIds::write);
        args.set(3, m);

        oneapi::internal::KernelRange range(n);

        ctx.run(range, kernel, args, &status);

        return status;
    }

    static services::Status transpose(const services::Buffer<algorithmFPType> & x, services::Buffer<algorithmFPType> & xt, const uint32_t n,
                                      const uint32_t p)
    {
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "transpose";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(4);
        args.set(0, x, oneapi::internal::AccessModeIds::read);
        args.set(1, xt, oneapi::internal::AccessModeIds::write);
        args.set(2, n);
        args.set(3, p);

        oneapi::internal::KernelRange range(n, p);

        ctx.run(range, kernel, args, &status);

        return services::Status();
    }

    static services::Status sumReduction(const services::Buffer<algorithmFPType> & reductionBuffer, const size_t nWorkGroups,
                                         algorithmFPType & result)
    {
        auto sumReductionArrayPtr      = reductionBuffer.toHost(data_management::readOnly);
        const auto * sumReductionArray = sumReductionArrayPtr.get();

        // Final summation with CPU
        for (size_t i = 0; i < nWorkGroups; i++)
        {
            result += sumReductionArray[i];
        }

        return services::Status();
    }

    // l1*||beta|| + l2*||beta||**2
    static services::Status regularization(const services::Buffer<algorithmFPType> & beta, const uint32_t nBeta, const uint32_t nClasses,
                                           algorithmFPType & reg, const algorithmFPType l1, const algorithmFPType l2)
    {
        services::Status status;
        const uint32_t n = nBeta * nClasses;

        const oneapi::internal::TypeIds::Id idType = oneapi::internal::TypeIds::id<algorithmFPType>();

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "regularization";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelNDRange range(1);

        // oneapi::internal::InfoDevice& info = ctx.getInfoDevice();
        // const size_t maxWorkItemSizes1d = info.max_work_item_sizes_1d;
        // const size_t maxWorkGroupSize = info.max_work_group_size;

        // TODO replace on min
        // size_t workItemsPerGroup = maxWorkItemSizes1d > maxWorkGroupSize ?
        //     maxWorkGroupSize : maxWorkItemSizes1d;

        size_t workItemsPerGroup = 256;

        const size_t nWorkGroups = getWorkgroupsCount(n, workItemsPerGroup);

        oneapi::internal::KernelRange localRange(workItemsPerGroup);
        oneapi::internal::KernelRange globalRange(workItemsPerGroup * nWorkGroups);

        range.local(localRange, &status);
        range.global(globalRange, &status);
        DAAL_CHECK_STATUS_VAR(status);

        oneapi::internal::UniversalBuffer buffer          = ctx.allocate(idType, nWorkGroups, &status);
        services::Buffer<algorithmFPType> reductionBuffer = buffer.get<algorithmFPType>();

        oneapi::internal::KernelArguments args(6 /*7*/);
        args.set(0, beta, oneapi::internal::AccessModeIds::read);
        args.set(1, nBeta);
        args.set(2, n);
        args.set(3, reductionBuffer, oneapi::internal::AccessModeIds::write);
        args.set(4, l1);
        args.set(5, l2);
        //args.set(6, oneapi::internal::LocalBuffer(idType, workItemsPerGroup));

        ctx.run(range, kernel, args, &status);

        DAAL_CHECK_STATUS(status, sumReduction(reductionBuffer, nWorkGroups, reg));

        return status;
    }

    // s1 + s2 + .. + sn
    static services::Status sum(const services::Buffer<algorithmFPType> & x, algorithmFPType & result, const uint32_t n)
    {
        services::Status status;
        const oneapi::internal::TypeIds::Id idType = oneapi::internal::TypeIds::id<algorithmFPType>();

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "sumReduction";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelNDRange range(1);

        // oneapi::internal::InfoDevice &info = ctx.getInfoDevice();
        // const size_t maxWorkItemSizes1d = info.max_work_item_sizes_1d;
        // const size_t maxWorkGroupSize = info.max_work_group_size;

        // TODO replace on min
        // size_t workItemsPerGroup = maxWorkItemSizes1d > maxWorkGroupSize ?
        //     maxWorkGroupSize : maxWorkItemSizes1d;

        size_t workItemsPerGroup = 256;

        const size_t nWorkGroups = getWorkgroupsCount(n, workItemsPerGroup);

        oneapi::internal::KernelRange localRange(workItemsPerGroup);
        oneapi::internal::KernelRange globalRange(workItemsPerGroup * nWorkGroups);

        range.local(localRange, &status);
        range.global(globalRange, &status);
        DAAL_CHECK_STATUS_VAR(status);

        oneapi::internal::UniversalBuffer buffer          = ctx.allocate(idType, nWorkGroups, &status);
        services::Buffer<algorithmFPType> reductionBuffer = buffer.get<algorithmFPType>();

        oneapi::internal::KernelArguments args(3 /*4*/);
        args.set(0, x, oneapi::internal::AccessModeIds::read);
        args.set(1, n);
        args.set(2, reductionBuffer, oneapi::internal::AccessModeIds::write);
        //args.set(3, oneapi::internal::LocalBuffer(idType, workItemsPerGroup));

        ctx.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS(status, sumReduction(reductionBuffer, nWorkGroups, result));

        return status;
    }

    // x = x + alpha
    // Where x - vector; alpha - scalar
    static services::Status addVectorScalar(services::Buffer<algorithmFPType> & x, const algorithmFPType alpha, const uint32_t n)
    {
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "addVectorScalar";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(2);
        args.set(0, x, oneapi::internal::AccessModeIds::write);
        args.set(1, alpha);

        oneapi::internal::KernelRange range(n);

        ctx.run(range, kernel, args, &status);

        return status;
    }

    // x = x + y[id]
    // Where x - vector; y - vector, id - index
    static services::Status addVectorScalar(services::Buffer<algorithmFPType> & x, const services::Buffer<algorithmFPType> & y, const uint32_t id,
                                            const uint32_t n)
    {
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "addVectorScalar2";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(3);
        args.set(0, x, oneapi::internal::AccessModeIds::write);
        args.set(1, y, oneapi::internal::AccessModeIds::read);
        args.set(2, id);

        oneapi::internal::KernelRange range(n);

        ctx.run(range, kernel, args, &status);

        return status;
    }

    static services::Status getXY(const services::Buffer<algorithmFPType> & xBuff, const services::Buffer<algorithmFPType> & yBuff,
                                  const services::Buffer<int> & indBuff, services::Buffer<algorithmFPType> aX, services::Buffer<algorithmFPType> aY,
                                  uint32_t nBatch, uint32_t p, bool interceptFlag)
    {
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const algorithmFPType interceptValue = interceptFlag ? algorithmFPType(1) : algorithmFPType(0);

        const char * const kernelName      = "getXY";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(7);
        args.set(0, xBuff, oneapi::internal::AccessModeIds::read);
        args.set(1, yBuff, oneapi::internal::AccessModeIds::read);
        args.set(2, indBuff, oneapi::internal::AccessModeIds::read);
        args.set(3, p);
        args.set(4, interceptValue);
        args.set(5, aX, oneapi::internal::AccessModeIds::write);
        args.set(6, aY, oneapi::internal::AccessModeIds::write);

        oneapi::internal::KernelRange range(p, nBatch);

        ctx.run(range, kernel, args, &status);

        return status;
    }

private:
    static void buildProgram(oneapi::internal::ClKernelFactoryIface & factory)
    {
        services::String options = oneapi::internal::getKeyFPType<algorithmFPType>();

        services::String cachekey("__daal_algorithms_optimization_solver_objective_function_");
        cachekey.add(options);

        options.add(" -D LOCAL_SUM_SIZE=256 "); //depends on workItemsPerGroup value

        factory.build(oneapi::internal::ExecutionTargetIds::device, cachekey.c_str(), clKernelObjectiveFunction, options.c_str());
    }
};

} // namespace internal
} // namespace objective_function
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal

#endif
