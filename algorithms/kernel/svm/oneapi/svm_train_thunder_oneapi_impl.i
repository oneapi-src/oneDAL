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
#include "service/kernel/data_management/service_numeric_table.h"
#include "service/kernel/service_utils.h"
#include "service/kernel/service_data_utils.h"
#include "externals/service_ittnotify.h"
#include "service/kernel/oneapi/blas_gpu.h"
#include "service/kernel/service_string_utils.h"
#include "algorithms/kernel/svm/oneapi/cl_kernels/svm_train_block_smo_oneapi.cl"

// TODO: DELETE
#include <cstdlib>
#include <chrono>
using namespace std::chrono;
//

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
template <typename algorithmFPType, typename ParameterType>
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, thunder>::updateGrad(const services::Buffer<algorithmFPType> & kernelWS,
                                                                                     const services::Buffer<algorithmFPType> & deltaalpha,
                                                                                     services::Buffer<algorithmFPType> & grad, const size_t nVectors,
                                                                                     const size_t nWS)
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
                                                                                 const algorithmFPType eps, const size_t nNoChanges,
                                                                                 int & sameLocalDiff)
{
    sameLocalDiff = abs(diff - diffPrev) < eps ? sameLocalDiff + 1 : 0;

    if (sameLocalDiff > nNoChanges)
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
services::Status SVMTrainOneAPI<algorithmFPType, ParameterType, thunder>::compute(const NumericTablePtr & xTable, NumericTable & yTable,
                                                                                  daal::algorithms::Model * r, const ParameterType * svmPar)
{
    services::Status status;

    auto & context       = services::Environment::getInstance()->getDefaultExecutionContext();
    const auto idType    = TypeIds::id<algorithmFPType>();
    const auto idTypeInt = TypeIds::id<int>();

    auto & deviceInfo = context.getInfoDevice();

    if (const char * env_p = std::getenv("SVM_VERBOSE"))
    {
        printf(">> VERBOSE MODE\n");
        verbose = true;
        printf(">> MAX WORK SIZE = %d\n", (int)deviceInfo.max_work_group_size);
    }

    const algorithmFPType C(svmPar->C);
    const algorithmFPType eps(svmPar->accuracyThreshold);
    const algorithmFPType tau(svmPar->tau);
    const size_t maxIterations(svmPar->maxIterations);
    const size_t cacheSize(svmPar->cacheSize);
    kernel_function::KernelIfacePtr kernel = svmPar->kernel->clone();

    const size_t nVectors  = xTable->getNumberOfRows();
    const size_t nFeatures = xTable->getNumberOfColumns();
    // ai = 0
    auto alphaU = context.allocate(idType, nVectors, &status);
    context.fill(alphaU, 0.0, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto alphaBuff = alphaU.get<algorithmFPType>();

    auto maskBuff = context.allocate(idType, nVectors, &status);

    // gradi = -yi
    auto gradU = context.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto gradBuff = gradU.get<algorithmFPType>();

    BlockDescriptor<algorithmFPType> yBD;
    DAAL_CHECK_STATUS(status, yTable.getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, yBD));
    auto yBuff = yBD.getBuffer();

    DAAL_CHECK_STATUS(status, Helper::initGrad(yBuff, gradBuff, nVectors));

    TaskWorkingSet<algorithmFPType> workSet(nVectors, verbose);

    DAAL_CHECK_STATUS(status, workSet.init());

    const size_t nWS        = workSet.getSize();
    const size_t nNoChanges = 5;
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
        printf(">>>> nVectors: %lu d: %lu nWS: %lu C: %f \n", nVectors, xTable->getNumberOfColumns(), nWS, C);
    }

    for (size_t iter = 0; iter < maxIterations; iter++)
    {
        if (iter != 0)
        {
            DAAL_CHECK_STATUS(status, workSet.saveWSIndeces());
        }
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

        const services::Buffer<int> & wsIndices = workSet.getWSIndeces();
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

        if (verbose)
        {
            printf(">>>> Kernel.compute\n");
            fflush(stdout);
        }

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

        if (checkStopCondition(diff, diffPrev, eps, nNoChanges, sameLocalDiff))
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
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
