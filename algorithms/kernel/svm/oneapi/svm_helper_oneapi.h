/* file: svm_helper_oneapi.h */
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

#ifndef __SVM_HELPER_ONEAPI_H__
#define __SVM_HELPER_ONEAPI_H__

#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/oneapi/sorter.h"
#include "service/kernel/oneapi/partition.h"
#include "externals/service_ittnotify.h"
#include "algorithms/kernel/svm/oneapi/cl_kernels/svm_kernels.cl"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace utils
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::oneapi::internal;

template <typename T>
inline const T min(const T a, const T b)
{
    return !(b < a) ? a : b;
}

template <typename T>
inline const T max(const T a, const T b)
{
    return (a < b) ? b : a;
}

template <typename T>
inline const T abs(const T & a)
{
    return a > 0 ? a : -a;
}

inline size_t maxpow2(size_t n)
{
    if (!(n & (n - 1)))
    {
        return n;
    }

    size_t count = 0;
    while (n > 1)
    {
        n >>= 1;
        count++;
    }
    return 1 << count;
}

template <typename algorithmFPType>
struct HelperSVM
{
    static services::Status buildProgram(ClKernelFactoryIface & factory)
    {
        services::String options = getKeyFPType<algorithmFPType>();

        services::String cachekey("__daal_algorithms_svm_");
        options.add(" -D LOCAL_SUM_SIZE=256 ");
        cachekey.add(options);

        services::Status status;
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSVM, options.c_str(), &status);
        return status;
    }

    static services::Status makeInversion(const services::Buffer<algorithmFPType> & x, services::Buffer<algorithmFPType> & res, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(makeInversion);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("makeInversion");

        KernelArguments args(2);
        args.set(0, x, AccessModeIds::read);
        args.set(1, res, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status makeRange(UniversalBuffer & x, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(makeRange);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("makeRange");

        KernelArguments args(1);
        args.set(0, x, AccessModeIds::readwrite);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status argSort(const UniversalBuffer & f, UniversalBuffer & values, UniversalBuffer & valuesBuf, UniversalBuffer & indecesSort,
                                    UniversalBuffer & indecesBuf, const size_t n)
    {
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        context.copy(values, 0, f, 0, n, &status);
        DAAL_CHECK_STATUS(status, makeRange(indecesSort, n));

        DAAL_CHECK_STATUS(status, sort::RadixSort::sortIndices(values, indecesSort, valuesBuf, indecesBuf, n));

        return status;
    }

    static services::Status copyDataByIndices(const services::Buffer<algorithmFPType> & x, const services::Buffer<uint32_t> & indX,
                                              services::Buffer<algorithmFPType> & newX, const size_t nWS, const uint32_t p)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(copyDataByIndices);
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "copyDataByIndices";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(5);
        args.set(0, x, oneapi::internal::AccessModeIds::read);
        args.set(1, indX, oneapi::internal::AccessModeIds::read);
        args.set(2, p);
        args.set(3, newX, oneapi::internal::AccessModeIds::write);

        oneapi::internal::KernelRange range(p, nWS);

        ctx.run(range, kernel, args, &status);

        return status;
    }

    static services::Status copyDataByIndices(const services::Buffer<algorithmFPType> & x, const services::Buffer<int32_t> & indX,
                                              services::Buffer<algorithmFPType> & newX, const size_t nWS, const uint32_t p)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(copyDataByIndices);
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "copyDataByIndicesInt";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(5);
        args.set(0, x, oneapi::internal::AccessModeIds::read);
        args.set(1, indX, oneapi::internal::AccessModeIds::read);
        args.set(2, p);
        args.set(3, newX, oneapi::internal::AccessModeIds::write);

        oneapi::internal::KernelRange range(p, nWS);

        ctx.run(range, kernel, args, &status);

        return status;
    }

    static services::Status checkUpper(const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & alpha,
                                       services::Buffer<uint32_t> & indicator, const algorithmFPType C, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkUpper);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkUpper");

        KernelArguments args(4);
        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicator, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status checkLower(const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & alpha,
                                       services::Buffer<uint32_t> & indicator, const algorithmFPType C, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkLower);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkLower");

        KernelArguments args(4);
        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicator, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status checkBorder(const services::Buffer<algorithmFPType> & alpha, services::Buffer<uint32_t> & mask, const algorithmFPType C,
                                        const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkBorder);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkBorder");

        KernelArguments args(3);
        args.set(0, alpha, AccessModeIds::read);
        args.set(1, C);
        args.set(2, mask, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status checkNonZeroBinary(const services::Buffer<algorithmFPType> & alpha, services::Buffer<uint32_t> & mask, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkNonZeroBinary);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkNonZeroBinary");

        KernelArguments args(2);
        args.set(0, alpha, AccessModeIds::read);
        args.set(1, mask, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status computeDualCoeffs(const services::Buffer<algorithmFPType> & y, services::Buffer<algorithmFPType> & alpha, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(computeDualCoeffs);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("computeDualCoeffs");

        KernelArguments args(2);
        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::readwrite);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }
};

} // namespace internal
} // namespace utils
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
