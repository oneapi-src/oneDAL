/* file: svm_train_boser_impl.i */
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

/*
//++
//  SVM training algorithm implementation
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  2. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  3. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_GPU_IMPL_I__
#define __SVM_TRAIN_GPU_IMPL_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_data_utils.h"

#include "algorithms/kernel/svm/oneapi/oneapi/cl_kernel/svm_train_oneapi.cl"

#include <cstdlib>

using namespace daal::internal;
using namespace daal::services::internal;

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
<<<<<<< HEAD
=======
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

template <typename algorithmFPType>
struct TaskWorkingSet
{
    TaskWorkingSet(size_t nVectors) : n(nVectors) {}

    services::Status init()
    {
        services::Status status;
        auto & context       = services::Environment::getInstance()->getDefaultExecutionContext();
        const auto idType    = TypeIds::id<algorithmFPType>();
        const auto idTypeInt = TypeIds::id<int>();
        sortedFIndices       = context.allocate(idTypeInt, nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);

        indicator = context.allocate(idTypeInt, nVectors, &status);
        context.fill(indicator, 0.0, &status);
        DAAL_CHECK_STATUS_VAR(status);

        fIndices = context.allocate(idTypeInt, nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);
        auto fIndicesBuf = fIndices<int>.get();
        DAAL_CHECK_STATUS(status, initIndex(fIndicesBuf, n));

        fIndices = context.allocate(idTypeInt, nVectors, &status);

        nWS = min(max_ws, n);
    }

    size_t getSize() const
    {
        constexpr size_t max_ws = 16;
        return min(max_ws, n);
    }

    services::Status argSort(services::Buffer<int> & fBuff)
    {
        services::Status status;
        ctx.copy(fIndices, 0, sortedFIndices, 0, n, &status);
        DAAL_CHECK_STATUS_VAR(status);
        auto sortedFIndicesBuff = sortedFIndices<int>.get();

        // TODO Replace radix sort
        {
            int * sortedFIndices_host = sortedFIndicesBuff.toHost(ReadWriteMode::readWrite).get();
            algorithmFPType * f_host  = fBuff.toHost(ReadWriteMode::read).get();
            std::sort(sortedFIndices_host, sortedFIndices_host + n, [](int i, int j) { return f_host[i] < f_host[j]; })
        }
        return status;
    }

    services::Status selectWS(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                              const services::Buffer<algorithmFPType> & fBuff, const algorithmFPType C)
    {
        services::Status status;
        static bool firstCall = true;
        const size_t q        = nWS / 2;

        if (firstCall)
        {
            firstCall = false;
        }
        else
        {
            // copy
        }

        DAAL_CHECK_STATUS(status, argSort(fBuff));
        auto sortedFIndicesBuff = sortedFIndices<int>.get();

        auto indicatorBuff = indicator<int>.get();

        DAAL_CHECK_STATUS(status, checkUpper());

        return status;
    }

    services::Buffer<int> & getSortedFIndices() const { return sortedFIndices<int>.get(); }

    services::Status checkUpper(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                                services::Buffer<int> & indicatorBuff, const algorithmFPType C)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkUpper);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkUpper");

        KernelArguments args(4);
        args.set(0, yBuff, AccessModeIds::read);
        args.set(1, alphaBuff, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicatorBuff, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    services::Status initIndex(services::Buffer<int> & x, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(range);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("range");

        KernelArguments args(1);
        args.set(0, x, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    size_t n;
    size_t nWS;

    UniversalBuffer sortedFIndices;
    UniversalBuffer indicator;
    UniversalBuffer fIndices;
}

using namespace daal::oneapi::internal;

>>>>>>> c13db2ed... ws add
template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, boser>::initGrad(const services::Buffer<algorithmFPType> & y, services::Buffer<algorithmFPType> & f,
                                                                  const size_t n)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(initGrad);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    services::Status status = buildProgram(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("initGradient");

    KernelArguments args(2);
    args.set(0, y, AccessModeIds::read);
    args.set(1, f, AccessModeIds::write);

    KernelRange range(n);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, boser>::compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r,
                                                                 const ParameterType * svmPar)
{
    services::Status status;

<<<<<<< HEAD
=======
    auto & context       = services::Environment::getInstance()->getDefaultExecutionContext();
    const auto idType    = TypeIds::id<algorithmFPType>();
    const auto idTypeInt = TypeIds::id<int>();

>>>>>>> c13db2ed... ws add
    if (const char * env_p = std::getenv("SVM_VERBOSE"))
    {
        printf(">> VERBOSE MODE\n");
        verbose = true;
    }

    const algorithmFPType C(svmPar.C);
    const algorithmFPType eps(svmPar.accuracyThreshold);
    const algorithmFPType tau(svmPar.tau);
    const size_t maxIterations(svmPar.maxIterations);
    // TODO
    const size_t innerMaxIterations(100);

    size_t nVectors = xTable->getNumberOfRows();

    // ai = 0
<<<<<<< HEAD
    UniversalBuffer alpha = ctx.allocate(idType, nVectors, &status);
    ctx.fill(alpha, 0.0, &status);
=======
    auto alphaU = context.allocate(idType, nVectors, &status);
    context.fill(alphaU, 0.0, &status);
>>>>>>> c13db2ed... ws add
    DAAL_CHECK_STATUS_VAR(status);
    auto alphaBuff = alphaU.get<algorithmFPType>();

    // fi = -yi
<<<<<<< HEAD
    UniversalBuffer f = ctx.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
    DAAL_CHECK_STATUS(status, initGrad(y, f, nVectors));

    UniversalBuffer alpha = ctx.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
=======
    auto fU = context.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto fBuff = fU.get<algorithmFPType>();
    {
        BlockDescriptor<algorithmFPType> yBD;
        yTable.getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, yBD);

        auto yBuff = yBD.getBuffer();
        DAAL_CHECK_STATUS(status, initGrad(yBuff, fBuff, nVectors));
        yTable.releaseBlockOfRows(yBD);
    }

    TaskWorkingSet workSet(nVectors);

    DAAL_CHECK_STATUS(status, workSet.init());
>>>>>>> c13db2ed... ws add

    const size_t nWS = workSet.getSize();

    if (verbose)
    {
        printf(">> LINE: %lu: nWS %lu\n", __LINE__, nWS);
    }

    // TODO transfer on GPU

    // for (size_t iter = 0; iter < maxIterations; i++)
    {
        if (verbose)
        {
            const auto t_0 = high_resolution_clock::now();
        }

<<<<<<< HEAD
        SelectWS();
=======
        DAAL_CHECK_STATUS(status, SelectWS(workSet));
>>>>>>> c13db2ed... ws add

        if (verbose)
        {
            const auto t_1           = high_resolution_clock::now();
            const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
            printf(">> SelectWS.compute time(ms) = %f\n", duration_sec);
        }
    }

    // return s.ok() ? task.setResultsToModel(*xTable, *static_cast<Model *>(r), svmPar->C) : s;
}

// inline Size MaxPow2(Size n) {
//     if (!(n & (n - 1))) {
//         return n;
//     }

//     Size count = 0;
//     while (n > 1) {
//         n >>= 1;
//         count++;
//     }
//     return 1 << count;
// }

template <typename algorithmFPType, typename ParameterType>
<<<<<<< HEAD
size_t SVMTrainOneAPI<boser, algorithmFPType, cpu>::SelectWorkingSetSize(const size_t n)
{
    // Depends on cache size
    // constexpr Size max_ws = 512;
    // constexpr Size max_ws = 1024;
    constexpr size_t max_ws = 256;
    // constexpr Size max_ws = 4096;
    return Min(max_ws, n);
    // return Min(MaxPow2(n_samples), max_ws);
=======
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::buildProgram(ClKernelFactoryIface & factory)
{
    services::String options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_svm_");
    cachekey.add(options);
    options.add(" -D LOCAL_SUM_SIZE=256 ");

    Status status;
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSVMTrain, options.c_str(), &status);
    return status;
>>>>>>> c13db2ed... ws add
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
