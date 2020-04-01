/* file: objective_function_utils_oneapi.h */
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

#ifndef __SVM_UTILS_ONEAPI_H__
#define __SVM_UTILS_ONEAPI_H__

#include "algorithms/kernel/svm/oneapi/cl_kernels/svm_train_oneapi.cl"
#include "service/kernel/data_management/service_numeric_table.h"

#include "externals/service_ittnotify.h"

// TODO: DELETE
#include <algorithm>
#include <cstdlib>
#include <chrono>
using namespace std::chrono;
//

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template <typename T>
inline const T & min(const T & a, const T & b)
{
    return !(b < a) ? a : b;
}

template <typename T>
inline const T & max(const T & a, const T & b)
{
    return (a < b) ? b : a;
}

template <typename T>
inline const T & abs(const T & a)
{
    return a > 0 ? a : -a;
}

using namespace daal::services::internal;
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
struct HelperSVM
{
    static services::Status buildProgram(ClKernelFactoryIface & factory)
    {
        services::String options = getKeyFPType<algorithmFPType>();

        services::String cachekey("__daal_algorithms_svm_");
        cachekey.add(options);
        options.add(" -D LOCAL_SUM_SIZE=256 ");

        services::Status status;
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSVMTrain, options.c_str(), &status);
        return status;
    }

    static services::Status range(services::Buffer<int> & x, const size_t nVectors)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(range);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("range");

        KernelArguments args(1);
        args.set(0, x, AccessModeIds::readwrite);

        KernelRange range(nVectors);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    static services::Status argSort(const services::Buffer<algorithmFPType> & fBuff, const services::Buffer<int> & fIndicesBuff,
                                    services::Buffer<int> & sortedFIndicesBuff, const size_t nVectors)
    {
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        context.copy(sortedFIndicesBuff, 0, fIndicesBuff, 0, nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);

        // TODO Replace radix sort
        {
            int * sortedFIndices_host = sortedFIndicesBuff.toHost(ReadWriteMode::readWrite).get();
            algorithmFPType * f_host  = fBuff.toHost(ReadWriteMode::readOnly).get();
            std::sort(sortedFIndices_host, sortedFIndices_host + nVectors, [=](int i, int j) { return f_host[i] < f_host[j]; });
        }
        return status;
    }

    static services::Status gatherIndices(const services::Buffer<int> & maskBuff, const services::Buffer<int> & xBuff, const size_t n,
                                          services::Buffer<int> & resBuff, size_t & nRes)
    {
        services::Status status;

        {
            int * indicator_host      = maskBuff.toHost(ReadWriteMode::readOnly).get();
            int * sortedFIndices_host = xBuff.toHost(ReadWriteMode::readOnly).get();
            int * tmpIndices_host     = resBuff.toHost(ReadWriteMode::readWrite).get();
            nRes                      = 0;
            for (int i = 0; i < n; i++)
            {
                if (indicator_host[i])
                {
                    tmpIndices_host[nRes++] = sortedFIndices_host[i];
                }
            }
        }
        return status;
    }

    static services::Status copyBlockIndices(const services::Buffer<algorithmFPType> & xBuff, const services::Buffer<int> & indBuff,
                                             services::Buffer<algorithmFPType> & newBuf, const uint32_t nWS, const uint32_t p)
    {
        services::Status status;

        oneapi::internal::ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
        oneapi::internal::ClKernelFactoryIface & factory = ctx.getClKernelFactory();

        buildProgram(factory);

        const char * const kernelName      = "copyBlockIndices";
        oneapi::internal::KernelPtr kernel = factory.getKernel(kernelName);

        oneapi::internal::KernelArguments args(4);
        args.set(0, xBuff, oneapi::internal::AccessModeIds::read);
        args.set(1, indBuff, oneapi::internal::AccessModeIds::read);
        args.set(2, p);
        args.set(3, newBuf, oneapi::internal::AccessModeIds::write);

        oneapi::internal::KernelRange range(p, nWS);

        ctx.run(range, kernel, args, &status);

        return status;
    }
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
