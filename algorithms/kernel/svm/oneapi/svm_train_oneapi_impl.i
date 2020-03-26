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
<<<<<<< HEAD

#include "algorithms/kernel/svm/oneapi/oneapi/cl_kernel/svm_train_oneapi.cl"

=======
#include "externals/service_ittnotify.h"
#include "algorithms/kernel/svm/oneapi/cl_kernels/svm_train_oneapi.cl"

// TODO: DELETE
#include <algorithm>
>>>>>>> 815734e6... fix build
#include <cstdlib>

DAAL_ITTNOTIFY_DOMAIN(svm_train.default.batch);

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::oneapi::internal;

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
    TaskWorkingSet(size_t nVectors, bool verbose) : nVectors(nVectors), verbose(verbose) {}

    services::Status init()
    {
        services::Status status;
        auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
        const auto idType = TypeIds::id<algorithmFPType>();

        sortedFIndices = context.allocate(TypeIds::id<int>(), nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);

        indicator = context.allocate(TypeIds::id<int>(), nVectors, &status);
        context.fill(indicator, 0.0, &status);
        DAAL_CHECK_STATUS_VAR(status);

        fIndices = context.allocate(TypeIds::id<int>(), nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);
        auto fIndicesBuf = fIndices.get<int>();
        DAAL_CHECK_STATUS(status, initIndex(fIndicesBuf, nVectors));

        fIndices = context.allocate(TypeIds::id<int>(), nVectors, &status);

        // TODO: Get from device info
        const size_t maxWS = 16;
        nWS = min(maxWS, nVectors);

        wsIndexes  = context.allocate(TypeIds::id<int>(), nWS, &status);
        tmpIndices = context.allocate(TypeIds::id<int>(), nVectors, &status);
        return status;
    }

    size_t getSize() const
    {
        return nWS;
    }

    services::Status argSort(const services::Buffer<algorithmFPType> & fBuff)
    {
        services::Status status;
        auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();

        context.copy(sortedFIndices, 0, fIndices, 0, nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);
        auto sortedFIndicesBuff = sortedFIndices.get<int>();

        // TODO Replace radix sort
        {
            int * sortedFIndices_host = sortedFIndicesBuff.toHost(ReadWriteMode::readWrite).get();
            algorithmFPType * f_host  = fBuff.toHost(ReadWriteMode::readOnly).get();
            std::sort(sortedFIndices_host, sortedFIndices_host + nVectors, [=](int i, int j) { return f_host[i] < f_host[j]; });
        }
        return status;
    }

    services::Status gatherIndices(size_t & nRes)
    {
        services::Status status;
        auto indicatorBuff      = indicator.get<int>();
        auto tmpIndicesBuff     = tmpIndices.get<int>();
        auto sortedFIndicesBuff = sortedFIndices.get<int>();

        {
            int * indicator_host      = indicatorBuff.toHost(ReadWriteMode::readOnly).get();
            int * sortedFIndices_host = sortedFIndicesBuff.toHost(ReadWriteMode::readOnly).get();
            int * tmpIndices_host     = tmpIndicesBuff.toHost(ReadWriteMode::readWrite).get();
            nRes                      = 0;
            for (int i = 0; i < nVectors; i++)
            {
                if (indicator_host[i])
                {
                    tmpIndices_host[nRes++] = sortedFIndices_host[i];
                }
            }
        }
        return status;
    }

    services::Status selectWS(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                              const services::Buffer<algorithmFPType> & fBuff, const algorithmFPType C)
    {
        auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();

        if (verbose)
        {
            printf(">>>> selectWS\n");
        }

        services::Status status;
        static bool firstCall = true;

        size_t nSelected = 0;

        if (firstCall)
        {
            firstCall = false;
        }
        else
        {
            // TODO!
            nSelected = 1;
            // copy
        }
        const size_t q = nWS - nSelected;

        DAAL_CHECK_STATUS(status, argSort(fBuff));
        auto sortedFIndicesBuff = sortedFIndices.get<int>();

        if (verbose)
        {
            printf(">> argSort: ");
            {
                int * sortedFIndices_host = sortedFIndicesBuff.toHost(ReadWriteMode::readOnly).get();
                for (int i = 0; i < min(16ul, nWS); i++)
                {
                    printf("%d ", sortedFIndices_host[i]);
                }
            }
            printf("\n");
            printf(">> sort val: ");
            {
                int * sortedFIndices_host = sortedFIndicesBuff.toHost(ReadWriteMode::readOnly).get();
                algorithmFPType * f_host = fBuff.toHost(ReadWriteMode::readOnly).get();
                for (int i = 0; i < min(16ul, nWS); i++)
                {
                    printf("%.2f ", f_host[sortedFIndices_host[i]]);
                }
            }
            printf("\n");
        }

        context.fill(indicator, 0.0, &status);
        DAAL_CHECK_STATUS_VAR(status);
        auto indicatorBuff = indicator.get<int>();

        {
            const size_t nSelect = q / 2;

            DAAL_CHECK_STATUS(status, checkUpper(yBuff, alphaBuff, indicatorBuff, C, nVectors));

            size_t selectUpper = 0;
            DAAL_CHECK_STATUS(status, gatherIndices(selectUpper));

            if (verbose)
            {
                printf(">> CheckUpper[tmpIndices] - %lu:  ", selectUpper);
                {
                    int * wsIndexes_host = tmpIndices.get<int>().toHost(ReadWriteMode::readOnly).get();
                    for (int i = 0; i < min(16ul, nWS); i++)
                    {
                        printf("%d ", wsIndexes_host[i]);
                    }
                }
                printf("\n");
            }

            const size_t nCopy = min(selectUpper, nSelect);

            context.copy(wsIndexes, nSelected, tmpIndices, 0, nCopy, &status);

            nSelected += nCopy;
        }

        {
            const size_t nSelect = nWS - nSelected;

            DAAL_CHECK_STATUS(status, checkLower(yBuff, alphaBuff, indicatorBuff, C, nVectors));

            size_t selectLower = 0;
            DAAL_CHECK_STATUS(status, gatherIndices(selectLower));

            if (verbose)
            {
                printf(">> checkLower[tmpIndices] - %lu:  ", selectLower);
                {
                    int * wsIndexes_host = tmpIndices.get<int>().toHost(ReadWriteMode::readOnly).get();
                    for (int i = 0; i < min(16ul, nWS); i++)
                    {
                        printf("%d ", wsIndexes_host[i]);
                    }
                }
                printf("\n");
            }

            const size_t nCopy = min(selectLower, nSelect);

            context.copy(wsIndexes, nSelected, tmpIndices, 0, nCopy, &status);

            nSelected += selectLower;
        }


        if (verbose)
        {
            printf(">> wsIndexes:  ");
            {
                int * wsIndexes_host = wsIndexes.get<int>().toHost(ReadWriteMode::readOnly).get();
                for (int i = 0; i < min(16ul, nWS); i++)
                {
                    printf("%d ", wsIndexes_host[i]);
                }
            }
            printf("\n");
        }


        return status;
    }

    services::Buffer<int> & getSortedFIndices() const { return sortedFIndices.get<int>(); }

    services::Status checkUpper(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                                services::Buffer<int> & indicatorBuff, const algorithmFPType C, const size_t nSelect)
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
        args.set(3, indicatorBuff, AccessModeIds::readwrite);

        KernelRange range(nSelect);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    services::Status checkLower(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                                services::Buffer<int> & indicatorBuff, const algorithmFPType C, const size_t nSelect)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkLower);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkLower");

        KernelArguments args(4);
        args.set(0, yBuff, AccessModeIds::read);
        args.set(1, alphaBuff, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicatorBuff, AccessModeIds::readwrite);

        KernelRange range(nSelect);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    services::Status initIndex(services::Buffer<int> & x, const size_t nVectors)
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

    services::Status buildProgram(ClKernelFactoryIface & factory)
    {
        services::String options = getKeyFPType<algorithmFPType>();

        services::String cachekey("__daal_algorithms_svm_");
        cachekey.add(options);
        options.add(" -D LOCAL_SUM_SIZE=256 ");

        Status status;
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelSVMTrain, options.c_str(), &status);
        return status;
    }

    size_t nVectors;
    size_t nWS;

    bool verbose;

    UniversalBuffer sortedFIndices;
    UniversalBuffer indicator;
    UniversalBuffer fIndices;
    UniversalBuffer wsIndexes;
    UniversalBuffer tmpIndices;
};

>>>>>>> c13db2ed... ws add
template <typename algorithmFPType, typename ParameterType>
<<<<<<< HEAD
services::Status SVMTrainOneAPI<algorithmFPType, boser>::initGrad(const services::Buffer<algorithmFPType> & y, services::Buffer<algorithmFPType> & f,
                                                                  const size_t n)
=======
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::initGrad(const services::Buffer<algorithmFPType> & y,
                                                                                 services::Buffer<algorithmFPType> & f, const size_t nVectors)
>>>>>>> 815734e6... fix build
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

    KernelRange range(nVectors);

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

    BlockDescriptor<algorithmFPType> yBD;
    yTable.getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, yBD);
    auto yBuff = yBD.getBuffer();
    DAAL_CHECK_STATUS(status, initGrad(yBuff, fBuff, nVectors));


    TaskWorkingSet<algorithmFPType> workSet(nVectors, verbose);

    DAAL_CHECK_STATUS(status, workSet.init());
>>>>>>> c13db2ed... ws add

    const size_t nWS = workSet.getSize();

    if (verbose)
    {
<<<<<<< HEAD
        printf(">> LINE: %lu: nWS %lu\n", __LINE__, nWS);
=======
        printf(">>>> nVectors: %lu d: %lu nWS: %lu C: %f \n", nVectors, xTable->getNumberOfColumns(), nWS, C);
>>>>>>> 815734e6... fix build
    }

    // TODO transfer on GPU

    // for (size_t iter = 0; iter < maxIterations; i++)
    {
        if (verbose)
        {
            const auto t_0 = high_resolution_clock::now();
        }

<<<<<<< HEAD
<<<<<<< HEAD
        SelectWS();
=======
        DAAL_CHECK_STATUS(status, SelectWS(workSet));
>>>>>>> c13db2ed... ws add
=======
        DAAL_CHECK_STATUS(status, workSet.selectWS(yBuff, alphaBuff, fBuff, C));
>>>>>>> 815734e6... fix build

        if (verbose)
        {
            const auto t_1           = high_resolution_clock::now();
            const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
            printf(">>>> SelectWS.compute time(ms) = %.1f\n", duration_sec);
        }
    }
<<<<<<< HEAD
=======

    DAAL_CHECK_STATUS(status, yTable.releaseBlockOfRows(yBD));

    return status;
>>>>>>> 815734e6... fix build

    // return s.ok() ? task.setResultsToModel(*xTable, *static_cast<Model *>(r), svmPar->C) : s;
}

// inline Size MaxPow2(Size nVectors) {
//     if (!(n & (n - 1))) {
//         return nVectors;
//     }

//     Size count = 0;
//     while (n > 1) {
//         nVectors >>= 1;
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
