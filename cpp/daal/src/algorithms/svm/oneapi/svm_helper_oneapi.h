/* file: svm_helper_oneapi.h */
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

#ifndef __SVM_HELPER_ONEAPI_H__
#define __SVM_HELPER_ONEAPI_H__

#include "src/data_management/service_numeric_table.h"
#include "src/sycl/sorter.h"
#include "src/sycl/partition.h"
#include "src/externals/service_profiler.h"
#include "src/algorithms/svm/oneapi/cl_kernels/svm_kernels.cl"
#include "src/services/service_data_utils.h"

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
using namespace daal::services::internal::sycl;

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
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSVM, options.c_str(), status);
        return status;
    }

    static services::Status makeInversion(const services::internal::Buffer<algorithmFPType> & x, services::internal::Buffer<algorithmFPType> & res,
                                          const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(makeInversion);

        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("makeInversion", status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(2, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(x.size() == n);
        DAAL_ASSERT(res.size() == n);

        args.set(0, x, AccessModeIds::read);
        args.set(1, res, AccessModeIds::write);
        KernelRange range(n);

        context.run(range, kernel, args, status);
        return status;
    }

    static services::Status makeRange(UniversalBuffer & x, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(makeRange);

        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("makeRange", status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(1, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, x, AccessModeIds::readwrite);

        KernelRange range(n);

        context.run(range, kernel, args, status);
        return status;
    }

    static services::Status argSort(const UniversalBuffer & f, UniversalBuffer & values, UniversalBuffer & valuesBuf, UniversalBuffer & indecesSort,
                                    UniversalBuffer & indecesBuf, const size_t n)
    {
        services::Status status;
        auto & context = services::internal::getDefaultContext();

        context.copy(values, 0, f, 0, n, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_CHECK_STATUS(status, makeRange(indecesSort, n));
        DAAL_CHECK_STATUS(status, sort::RadixSort::sortIndices(values, indecesSort, valuesBuf, indecesBuf, n));
        return status;
    }

    static services::Status copyDataByIndices(const services::internal::Buffer<algorithmFPType> & x,
                                              const services::internal::Buffer<uint32_t> & indX, services::internal::Buffer<algorithmFPType> & newX,
                                              const size_t nWS, const size_t p)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(copyDataByIndices);
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName              = "copyDataByIndices";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(indX.size() == nWS);
        DAAL_ASSERT(newX.size() == nWS * p);

        args.set(0, x, services::internal::sycl::AccessModeIds::read);
        args.set(1, indX, services::internal::sycl::AccessModeIds::read);
        DAAL_ASSERT(p <= uint32max);
        args.set(2, static_cast<uint32_t>(p));
        args.set(3, newX, services::internal::sycl::AccessModeIds::write);

        services::internal::sycl::KernelRange range(p, nWS);

        ctx.run(range, kernel, args, status);
        return status;
    }

    static services::Status copyDataByIndices(const services::internal::Buffer<algorithmFPType> & x, const services::internal::Buffer<int> & indX,
                                              services::internal::Buffer<algorithmFPType> & newX, const size_t nWS, const size_t p)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(copyDataByIndices);
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName              = "copyDataByIndicesInt";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(indX.size() == nWS);
        DAAL_ASSERT(newX.size() == nWS * p);

        args.set(0, x, services::internal::sycl::AccessModeIds::read);
        args.set(1, indX, services::internal::sycl::AccessModeIds::read);
        DAAL_ASSERT(p <= uint32max);
        args.set(2, static_cast<int32_t>(p));
        args.set(3, newX, services::internal::sycl::AccessModeIds::write);
        services::internal::sycl::KernelRange range(p, nWS);

        ctx.run(range, kernel, args, status);
        return status;
    }

    static services::Status copyRowIndicesByIndices(const services::internal::Buffer<size_t> & rowsIn, const UniversalBuffer & ind,
                                                    services::internal::Buffer<size_t> & rowsOut, const size_t nWS, size_t & dataSize)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(copyCSRByIndices);
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName              = "copyRowIndicesByIndices";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(5, status);
        DAAL_CHECK_STATUS_VAR(status);

        auto dataSizeU = ctx.allocate(TypeIds::id<size_t>(), 1, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(rowsOut.size() == nWS + 1);
        DAAL_ASSERT(rowsIn.size() > nWS);

        args.set(0, rowsIn, services::internal::sycl::AccessModeIds::read);
        args.set(1, ind, services::internal::sycl::AccessModeIds::read);
        args.set(2, rowsOut, services::internal::sycl::AccessModeIds::write);
        args.set(3, nWS);
        args.set(4, dataSizeU);

        services::internal::sycl::KernelRange range(1);

        ctx.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);

        auto svValuesHosrPtr = dataSizeU.get<size_t>().toHost(data_management::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        dataSize = *svValuesHosrPtr;

        return status;
    }

    static services::Status copyCSRByIndices(const services::internal::Buffer<size_t> & rowsIn, const services::internal::Buffer<size_t> & rowsOut,
                                             const UniversalBuffer & ind, const services::internal::Buffer<algorithmFPType> & val,
                                             const services::internal::Buffer<size_t> & cols, services::internal::Buffer<algorithmFPType> & valOut,
                                             services::internal::Buffer<size_t> & colsOut, const size_t nWS, const size_t p)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(copyCSRByIndices);
        services::Status status;

        services::internal::sycl::ExecutionContextIface & ctx    = services::internal::getDefaultContext();
        services::internal::sycl::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName              = "copyCSRByIndices";
        services::internal::sycl::KernelPtr kernel = factory.getKernel(kernelName, status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::sycl::KernelArguments args(7, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(rowsOut.size() == nWS + 1);
        DAAL_ASSERT(rowsIn.size() > nWS);

        args.set(0, rowsIn, services::internal::sycl::AccessModeIds::read);
        args.set(1, rowsOut, services::internal::sycl::AccessModeIds::read);
        args.set(2, ind, services::internal::sycl::AccessModeIds::read);
        args.set(3, val, services::internal::sycl::AccessModeIds::read);
        args.set(4, cols, services::internal::sycl::AccessModeIds::read);
        args.set(5, valOut, services::internal::sycl::AccessModeIds::write);
        args.set(6, colsOut, services::internal::sycl::AccessModeIds::write);

        services::internal::sycl::KernelRange range(nWS, p);

        ctx.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
        return status;
    }

    static services::Status checkUpper(const services::internal::Buffer<algorithmFPType> & y,
                                       const services::internal::Buffer<algorithmFPType> & alpha, services::internal::Buffer<uint32_t> & indicator,
                                       const algorithmFPType C, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkUpper);

        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkUpper", status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(y.size() == n);
        DAAL_ASSERT(alpha.size() == n);
        DAAL_ASSERT(indicator.size() == n);

        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicator, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, status);
        return status;
    }

    static services::Status checkLower(const services::internal::Buffer<algorithmFPType> & y,
                                       const services::internal::Buffer<algorithmFPType> & alpha, services::internal::Buffer<uint32_t> & indicator,
                                       const algorithmFPType C, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkLower);

        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkLower", status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(y.size() == n);
        DAAL_ASSERT(alpha.size() == n);
        DAAL_ASSERT(indicator.size() == n);

        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicator, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, status);
        return status;
    }

    static services::Status checkBorder(const services::internal::Buffer<algorithmFPType> & alpha, services::internal::Buffer<uint32_t> & mask,
                                        const algorithmFPType C, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkBorder);

        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkBorder", status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(alpha.size() == n);
        DAAL_ASSERT(mask.size() == n);

        args.set(0, alpha, AccessModeIds::read);
        args.set(1, C);
        args.set(2, mask, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, status);
        return status;
    }

    static services::Status checkNonZeroBinary(const services::internal::Buffer<algorithmFPType> & alpha, services::internal::Buffer<uint32_t> & mask,
                                               const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkNonZeroBinary);

        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkNonZeroBinary", status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(2, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(alpha.size() == n);
        DAAL_ASSERT(mask.size() == n);

        args.set(0, alpha, AccessModeIds::read);
        args.set(1, mask, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, status);
        return status;
    }

    static services::Status computeDualCoeffs(const services::internal::Buffer<algorithmFPType> & y,
                                              services::internal::Buffer<algorithmFPType> & alpha, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(computeDualCoeffs);

        auto & context = services::internal::getDefaultContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("computeDualCoeffs", status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(2, status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_ASSERT(y.size() == n);
        DAAL_ASSERT(alpha.size() == n);

        args.set(0, y, AccessModeIds::read);
        args.set(1, alpha, AccessModeIds::readwrite);

        KernelRange range(n);

        context.run(range, kernel, args, status);
        return status;
    }

private:
    static constexpr size_t uint32max = static_cast<size_t>(services::internal::MaxVal<uint32_t>::get());
};

} // namespace internal
} // namespace utils
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
