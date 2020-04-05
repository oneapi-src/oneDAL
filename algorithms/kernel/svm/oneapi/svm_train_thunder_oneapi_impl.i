/* file: svm_train_thunder_oneapi_impl.i */
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

//  1. Zeyi Wen, Jiashuai Shi, Bingsheng He
//     ThunderSVM: A Fast SVM Library on GPUs and CPUs,
//     Journal of Machine Learning Research, 19, 1-5 (2018)
//  2. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  3. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  4. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_THUNDER_ONEAPI_IMPL_I__
#define __SVM_TRAIN_THUNDER_ONEAPI_IMPL_I__

#include "externals/service_memory.h"
#include "service/kernel/data_management/service_micro_table.h"
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_data_utils.h"
<<<<<<< HEAD

#include "algorithms/kernel/svm/oneapi/oneapi/cl_kernel/svm_train_oneapi.cl"

=======
#include "externals/service_ittnotify.h"
#include "service/kernel/oneapi/blas_gpu.h"
#include "service/kernel/service_string_utils.h"
#include "algorithms/kernel/svm/oneapi/cl_kernels/svm_train_block_smo_oneapi.cl"

// TODO: DELETE
#include <algorithm>
>>>>>>> 815734e6... fix build
#include <cstdlib>

#include "algorithms/kernel/svm/oneapi/svm_train_cache_oneapi.h"
#include "algorithms/kernel/svm/oneapi/svm_train_workset_oneapi.h"
#include "algorithms/kernel/svm/oneapi/svm_train_result_oneapi.h"

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
<<<<<<< HEAD
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
=======
>>>>>>> 14431dac... kernel support was added

>>>>>>> c13db2ed... ws add
=======
>>>>>>> b7022f17... add linear kernel
template <typename algorithmFPType, typename ParameterType>
<<<<<<< HEAD:algorithms/kernel/svm/oneapi/svm_train_oneapi_impl.i
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

    services::Status status = Helper::buildProgram(factory);
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
<<<<<<< HEAD
services::Status SVMTrainOneAPI<algorithmFPType, boser>::compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r,
                                                                 const ParameterType * svmPar)
=======
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::updateGrad(const services::Buffer<algorithmFPType> & kernelWS,
                                                                                   const services::Buffer<algorithmFPType> & deltaalpha,
                                                                                   services::Buffer<algorithmFPType> & grad, const size_t nVectors,
                                                                                   const size_t nWS)
=======
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, thunder>::updateGrad(const services::Buffer<algorithmFPType> & kernelWS,
                                                                                     const services::Buffer<algorithmFPType> & deltaalpha,
                                                                                     services::Buffer<algorithmFPType> & grad, const size_t nVectors,
                                                                                     const size_t nWS)
>>>>>>> c8ef0452... stabile and working version:algorithms/kernel/svm/oneapi/svm_train_thunder_oneapi_impl.i
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateGrad);
    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, nVectors, 1, nWS,
                                           algorithmFPType(1), kernelWS, nVectors, 0, deltaalpha, 1, 0, algorithmFPType(1), grad, 1, 0);
}

template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, thunder>::smoKernel(
    const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & kernelWsRows, const services::Buffer<int> & wsIndices,
    const int ldK, const services::Buffer<algorithmFPType> & f, const algorithmFPType C, const algorithmFPType eps, const algorithmFPType tau,
    const int maxInnerIteration, services::Buffer<algorithmFPType> & alpha, services::Buffer<algorithmFPType> & deltaalpha,
    services::Buffer<algorithmFPType> & resinfo, const size_t nWS)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(smoKernel);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    services::String build_options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_svm_smo_block_");
    cachekey.add(build_options);
    build_options.add(" -D WS_SIZE=");
    char WsString[60];
    services::internal::toStringBuffer<int>(nWS, WsString);
    build_options.add(WsString);

    services::Status status;
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelBlockSMO, build_options.c_str(), &status);

    DAAL_CHECK_STATUS_VAR(status);

    auto kernel = factory.getKernel("smoKernel");

    KernelArguments args(12);
    args.set(0, y, AccessModeIds::read);
    args.set(1, kernelWsRows, AccessModeIds::read);
    args.set(2, wsIndices, AccessModeIds::read);
    args.set(3, ldK);
    args.set(4, f, AccessModeIds::read);
    args.set(5, C);
    args.set(6, eps);
    args.set(7, tau);
    args.set(8, maxInnerIteration);
    args.set(9, alpha, AccessModeIds::readwrite);
    args.set(10, deltaalpha, AccessModeIds::readwrite);
    args.set(11, resinfo, AccessModeIds::readwrite);

    KernelRange localRange(nWS);
    KernelRange globalRange(nWS);

    KernelNDRange range(1);
    range.global(globalRange, &status);
    DAAL_CHECK_STATUS_VAR(status);
    range.local(localRange, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.run(range, kernel, args, &status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

template <typename algorithmFPType, typename ParameterType>
bool SVMTrainOneAPI<algorithmFPType, ParameterType, thunder>::checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev,
                                                                                 const algorithmFPType eps, int & sameLocalDiff)
{
    sameLocalDiff = abs(diff - diffPrev) < eps ? sameLocalDiff + 1 : 0;

    if (sameLocalDiff > 5)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType, typename ParameterType>
double SVMTrainOneAPI<algorithmFPType, ParameterType, thunder>::calculateObjective(const services::Buffer<algorithmFPType> & y,
                                                                                   const services::Buffer<algorithmFPType> & alpha,
                                                                                   const services::Buffer<algorithmFPType> & grad,
                                                                                   const size_t nVectors)
{
    double obj     = 0.0f;
    auto yHost     = y.toHost(ReadWriteMode::readOnly).get();
    auto alphaHost = alpha.toHost(ReadWriteMode::readOnly).get();
    auto gradHost  = grad.toHost(ReadWriteMode::readOnly).get();
    for (size_t i = 0; i < nVectors; i++)
    {
        obj += alphaHost[i] - (gradHost[i] + yHost[i]) * alphaHost[i] * yHost[i] * 0.5;
    }
    return -obj;
}

template <typename algorithmFPType, typename ParameterType>
<<<<<<< HEAD:algorithms/kernel/svm/oneapi/svm_train_oneapi_impl.i
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, boser>::compute(const NumericTablePtr & xTable, NumericTable & yTable,
                                                                                daal::algorithms::Model * r, const ParameterType * svmPar)
>>>>>>> 64f30ec0... smo local add & update F
=======
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, thunder>::compute(const NumericTablePtr & xTable, NumericTable & yTable,
                                                                                  daal::algorithms::Model * r, const ParameterType * svmPar)
>>>>>>> c8ef0452... stabile and working version:algorithms/kernel/svm/oneapi/svm_train_thunder_oneapi_impl.i
{
    services::Status status;

<<<<<<< HEAD
=======
    auto & context       = services::Environment::getInstance()->getDefaultExecutionContext();
    const auto idType    = TypeIds::id<algorithmFPType>();
    const auto idTypeInt = TypeIds::id<int>();

<<<<<<< HEAD
>>>>>>> c13db2ed... ws add
=======
    auto & deviceInfo = context.getInfoDevice();

>>>>>>> 64f30ec0... smo local add & update F
    if (const char * env_p = std::getenv("SVM_VERBOSE"))
    {
        printf(">> VERBOSE MODE\n");
        verbose = true;
        printf(">> MAX WORK SIZE = %d\n", (int)deviceInfo.max_work_group_size);
    }

<<<<<<< HEAD
    const algorithmFPType C(svmPar.C);
    const algorithmFPType eps(svmPar.accuracyThreshold);
    const algorithmFPType tau(svmPar.tau);
    const size_t maxIterations(svmPar.maxIterations);
=======
    const algorithmFPType C(svmPar->C);
    const algorithmFPType eps(svmPar->accuracyThreshold);
    const algorithmFPType tau(svmPar->tau);
    const size_t maxIterations(svmPar->maxIterations);
    const size_t cacheSize(svmPar->cacheSize);
<<<<<<< HEAD
    kernel_function::KernelIfacePtr kernel = svmPar.kernel->clone();
>>>>>>> 14431dac... kernel support was added
    // TODO
    const size_t innerMaxIterations(100);

    size_t nVectors = xTable->getNumberOfRows();

=======
    kernel_function::KernelIfacePtr kernel = svmPar->kernel->clone();

    const size_t nVectors  = xTable->getNumberOfRows();
    const size_t nFeatures = xTable->getNumberOfColumns();
>>>>>>> 53c7b11f... fix bugs
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

<<<<<<< HEAD:algorithms/kernel/svm/oneapi/svm_train_oneapi_impl.i
    // fi = -yi
<<<<<<< HEAD
    UniversalBuffer f = ctx.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
    DAAL_CHECK_STATUS(status, initGrad(y, f, nVectors));

    UniversalBuffer alpha = ctx.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
=======
    auto fU = context.allocate(idType, nVectors, &status);
=======
    // gradi = -yi
    auto gradU = context.allocate(idType, nVectors, &status);
>>>>>>> c8ef0452... stabile and working version:algorithms/kernel/svm/oneapi/svm_train_thunder_oneapi_impl.i
    DAAL_CHECK_STATUS_VAR(status);
    auto gradBuff = gradU.get<algorithmFPType>();

    BlockDescriptor<algorithmFPType> yBD;
    DAAL_CHECK_STATUS(status, yTable.getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, yBD));
    auto yBuff = yBD.getBuffer();

    DAAL_CHECK_STATUS(status, Helper::initGrad(yBuff, gradBuff, nVectors));

    TaskWorkingSet<algorithmFPType> workSet(nVectors, verbose);

    DAAL_CHECK_STATUS(status, workSet.init());
>>>>>>> c13db2ed... ws add

    const size_t nWS = workSet.getSize();
    const size_t innerMaxIterations(nWS * 1000);

    auto deltaalphaU = context.allocate(idType, nWS, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto deltaalphaBuff = deltaalphaU.get<algorithmFPType>();

    auto resinfoU = context.allocate(idType, 2, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto resinfoBuff = resinfoU.get<algorithmFPType>();

    int localInnerIteration  = 0;
    int sameLocalDiff        = 0;
    int innerIteration       = -1;
    algorithmFPType diff     = algorithmFPType(0);
    algorithmFPType diffPrev = algorithmFPType(0);

    SVMCacheOneAPIIface<algorithmFPType> * cache = nullptr;

    float ws_select      = 0.0;
    float kernel_compute = 0.0;
    float solver         = 0.0;
    float update_grad    = 0.0;

    if (cacheSize > nWS * nVectors * sizeof(algorithmFPType))
    {
        // TODO!
        cache = SVMCacheOneAPI<noCache, algorithmFPType>::create(cacheSize, nWS, nVectors, xTable, kernel, verbose, status);
    }
    else
    {
        cache = SVMCacheOneAPI<noCache, algorithmFPType>::create(cacheSize, nWS, nVectors, xTable, kernel, verbose, status);
    }

    if (verbose)
    {
<<<<<<< HEAD
        printf(">> LINE: %lu: nWS %lu\n", __LINE__, nWS);
=======
        printf(">>>> nVectors: %lu d: %lu nWS: %lu C: %f \n", nVectors, xTable->getNumberOfColumns(), nWS, C);
>>>>>>> 815734e6... fix build
    }

    // TODO transfer on GPU

    for (size_t iter = 0; iter < maxIterations; iter++)
    {
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
        if (iter != 0)
        {
            DAAL_CHECK_STATUS(status, workSet.saveWSIndeces());
        }
>>>>>>> 2b6b048e... add res model
        {
            const auto t_0 = high_resolution_clock::now();

            DAAL_CHECK_STATUS(status, workSet.selectWS(yBuff, alphaBuff, gradBuff, C));

            if (verbose)
            {
                const auto t_1           = high_resolution_clock::now();
                const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
                printf(">>>> SelectWS.compute time(ms) = %.1f\n", duration_sec);
                ws_select += duration_sec;
                fflush(stdout);
            }
        }

<<<<<<< HEAD
        auto & wsIndices = workSet.getWSIndeces();
<<<<<<< HEAD
>>>>>>> 64f30ec0... smo local add & update F

=======
>>>>>>> ca30c048... add sort
        {
            const auto t_0 = high_resolution_clock::now();

            DAAL_CHECK_STATUS(status, cache->compute(xBuff, wsIndices, nFeatures));
=======
        const services::Buffer<int> & wsIndices = workSet.getWSIndeces();
<<<<<<< HEAD:algorithms/kernel/svm/oneapi/svm_train_oneapi_impl.i
        DAAL_CHECK_STATUS(status, cache->compute(xTable, wsIndices, nFeatures));
>>>>>>> 8a074e5f... workin training
=======
        {
            const auto t_0 = high_resolution_clock::now();
            DAAL_CHECK_STATUS(status, cache->compute(xTable, wsIndices, nFeatures));
            if (verbose)
            {
                const auto t_1           = high_resolution_clock::now();
                const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
                printf(">>>> kerel.compute time(ms) = %.1f\n", duration_sec);
                kernel_compute += duration_sec;
                fflush(stdout);
            }
        }
>>>>>>> c8ef0452... stabile and working version:algorithms/kernel/svm/oneapi/svm_train_thunder_oneapi_impl.i

        if (verbose)
        {
            printf(">>>> Kernel.compute\n");
            fflush(stdout);
        }
<<<<<<< HEAD
    }
<<<<<<< HEAD
=======
=======
>>>>>>> 64f30ec0... smo local add & update F

        // TODO: Save half elements from kernel on 1+ iterations
        const services::Buffer<algorithmFPType> & kernelWS = cache->getSetRowsBlock();

        {
            const auto t_0 = high_resolution_clock::now();

            DAAL_CHECK_STATUS(status, smoKernel(yBuff, kernelWS, wsIndices, nVectors, gradBuff, C, eps, tau, innerMaxIterations, alphaBuff,
                                                deltaalphaBuff, resinfoBuff, nWS));
            {
                auto resinfoHost = resinfoBuff.toHost(ReadWriteMode::readOnly, &status).get();
                innerIteration   = int(resinfoHost[0]);
                diff             = resinfoHost[1];
                localInnerIteration += innerIteration;
            }

            if (verbose)
            {
                const auto t_1           = high_resolution_clock::now();
                const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
                printf(">>>> smoKernel (ms) = %.3f\n", duration_sec);
                printf(">>>> iter %lu localInnerIteration % d innerIteration = %d diff = %.1f\n", iter, localInnerIteration, innerIteration, diff);
                solver += duration_sec;
                fflush(stdout);
            }
        }

        {
            const auto t_0 = high_resolution_clock::now();

            DAAL_CHECK_STATUS(status, updateGrad(kernelWS, deltaalphaBuff, gradBuff, nVectors, nWS));

            if (verbose)
            {
                const auto t_1           = high_resolution_clock::now();
                const float duration_sec = duration_cast<milliseconds>(t_1 - t_0).count();
                printf(">>>> updateGrad (ms) = %.1f\n", duration_sec);
                fflush(stdout);
                update_grad += duration_sec;
            }
        }
        if (verbose)
        {
            double obj = calculateObjective(yBuff, alphaBuff, gradBuff, nVectors);
            printf(">>>>>> calculateObjective obj = %.3lf\n", obj);
        }

        if (checkStopCondition(diff, diffPrev, eps, sameLocalDiff))
        {
            if (verbose)
            {
                printf(">>>> checkStopCondition diff = %.3f diffPrev = %.3f\n", diff, diffPrev);
            }
            break;
        }
        diffPrev = diff;
    }

    if (verbose)
    {
        printf(">>>>>> SELECT WS (ms) %.3lf\n", ws_select);
        printf(">>>>>> KERNEL COMPUTE (ms) %.3lf\n", kernel_compute);
        printf(">>>>>> SMO (ms) %.3lf\n", solver);
        printf(">>>>>> UPDATE GRAD WS (ms) %.3lf\n", update_grad);
    }

    SaveResultModel<algorithmFPType> result(alphaBuff, gradBuff, yBuff, C, nVectors);

    DAAL_CHECK_STATUS(status, result.init());
    DAAL_CHECK_STATUS(status, result.setResultsToModel(xTable, *static_cast<Model *>(r)));

    DAAL_CHECK_STATUS(status, yTable.releaseBlockOfRows(yBD));

    delete cache;

    return status;
<<<<<<< HEAD
>>>>>>> 815734e6... fix build

    // return s.ok() ? task.setResultsToModel(*xTable, *static_cast<Model *>(r), svmPar->C) : s;
=======
>>>>>>> 2b6b048e... add res model
}

<<<<<<< HEAD
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

<<<<<<< HEAD
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

=======
>>>>>>> 14431dac... kernel support was added
=======
>>>>>>> 29a8c4e9... add prediction
} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
