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
//  SVM training algorithm implementation thunder method
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

#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"
#include "src/sycl/blas_gpu.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_ittnotify.h"
#include "src/externals/service_service.h"
#include "src/algorithms/svm/oneapi/cl_kernels/svm_train_block_smo_oneapi.cl"

#include "src/algorithms/svm/oneapi/svm_train_cache_oneapi.h"
#include "src/algorithms/svm/oneapi/svm_train_workset_oneapi.h"
#include "src/algorithms/svm/oneapi/svm_train_result_oneapi.h"

DAAL_ITTNOTIFY_DOMAIN(svm_train.default.batch);

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
using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
services::Status SVMTrainOneAPI<algorithmFPType, thunder>::updateGrad(const services::Buffer<algorithmFPType> & kernelWS,
                                                                      const services::Buffer<algorithmFPType> & deltaalpha,
                                                                      services::Buffer<algorithmFPType> & grad, const size_t nVectors,
                                                                      const size_t nWS)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(updateGrad);
    return BlasGpu<algorithmFPType>::xgemm(math::Layout::RowMajor, math::Transpose::Trans, math::Transpose::NoTrans, nVectors, 1, nWS,
                                           algorithmFPType(1), kernelWS, nVectors, 0, deltaalpha, 1, 0, algorithmFPType(1), grad, 1, 0);
}

template <typename algorithmFPType>
services::Status SVMTrainOneAPI<algorithmFPType, thunder>::smoKernel(
    const services::Buffer<algorithmFPType> & y, const services::Buffer<algorithmFPType> & kernelWsRows, const services::Buffer<uint32_t> & wsIndices,
    const uint32_t ldK, const services::Buffer<algorithmFPType> & f, const algorithmFPType C, const algorithmFPType accuracyThreshold,
    const algorithmFPType tau, const uint32_t maxInnerIteration, services::Buffer<algorithmFPType> & alpha,
    services::Buffer<algorithmFPType> & deltaalpha, services::Buffer<algorithmFPType> & resinfo, const size_t nWS)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(smoKernel);

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & factory = context.getClKernelFactory();

    services::String build_options = getKeyFPType<algorithmFPType>();

    services::String cachekey("__daal_algorithms_svm_smo_block_");
    build_options.add(" -D WS_SIZE=");
    char bufferString[DAAL_MAX_STRING_SIZE] = { 0 };
    services::daal_int_to_string(bufferString, DAAL_MAX_STRING_SIZE, int(nWS));
    build_options.add(bufferString);
    build_options.add(" -D SIMD_WIDTH=64 ");
    cachekey.add(build_options);

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
    args.set(6, accuracyThreshold);
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

template <typename algorithmFPType>
bool SVMTrainOneAPI<algorithmFPType, thunder>::checkStopCondition(const algorithmFPType diff, const algorithmFPType diffPrev,
                                                                  const algorithmFPType accuracyThreshold, size_t & sameLocalDiff)
{
    sameLocalDiff = utils::internal::abs(diff - diffPrev) < accuracyThreshold * 1e-2 ? sameLocalDiff + 1 : 0;

    if (sameLocalDiff > nNoChanges || diff < accuracyThreshold)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType>
services::Status SVMTrainOneAPI<algorithmFPType, thunder>::compute(const NumericTablePtr & xTable, NumericTable & yTable, daal::algorithms::Model * r,
                                                                   const ParameterType * svmPar)
{
    services::Status status;

    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    const auto idType = TypeIds::id<algorithmFPType>();

    const algorithmFPType C(svmPar->C);
    const algorithmFPType accuracyThreshold(svmPar->accuracyThreshold);
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
    auto alphaBuff = alphaU.template get<algorithmFPType>();

    BlockDescriptor<algorithmFPType> yBD;
    DAAL_CHECK_STATUS(status, yTable.getBlockOfRows(0, nVectors, ReadWriteMode::readOnly, yBD));
    auto yBuff = yBD.getBuffer();

    // gradi = -yi
    auto gradU = context.allocate(idType, nVectors, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto gradBuff = gradU.template get<algorithmFPType>();

    DAAL_CHECK_STATUS(status, Helper::makeInversion(yBuff, gradBuff, nVectors));

    TaskWorkingSet<algorithmFPType> workSet(nVectors);

    DAAL_CHECK_STATUS(status, workSet.init());

    const size_t nWS = workSet.getSize();

    const size_t innerMaxIterations(nWS * cInnerIterations);

    auto deltaalphaU = context.allocate(idType, nWS, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto deltaalphaBuff = deltaalphaU.template get<algorithmFPType>();

    auto resinfoU = context.allocate(idType, 2, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto resinfoBuff = resinfoU.template get<algorithmFPType>();

    algorithmFPType diff     = algorithmFPType(0);
    algorithmFPType diffPrev = algorithmFPType(0);

    size_t innerIteration = 0;
    size_t sameLocalDiff  = 0;

    SVMCacheOneAPIPtr<algorithmFPType> cachePtr;

    if (cacheSize >= nVectors * nVectors * sizeof(algorithmFPType))
    {
        // TODO: support the simple cache for thunder method
        cachePtr = SVMCacheOneAPI<noCache, algorithmFPType>::create(cacheSize, nWS, nVectors, xTable, kernel, status);
    }
    else
    {
        cachePtr = SVMCacheOneAPI<noCache, algorithmFPType>::create(cacheSize, nWS, nVectors, xTable, kernel, status);
    }

    size_t iter = 0;
    for (; iter < maxIterations; iter++)
    {
        if (iter != 0)
        {
            DAAL_CHECK_STATUS(status, workSet.copyLastToFirst());
            DAAL_CHECK_STATUS(status, cachePtr->copyLastToFirst());
        }

        DAAL_CHECK_STATUS(status, workSet.selectWS(yBuff, alphaBuff, gradBuff, C));

        const services::Buffer<uint32_t> & wsIndices = workSet.getWSIndeces();
        DAAL_CHECK_STATUS(status, cachePtr->compute(xTable, wsIndices, nFeatures));

        const services::Buffer<algorithmFPType> & kernelWS = cachePtr->getRowsBlock();

        DAAL_CHECK_STATUS(status, smoKernel(yBuff, kernelWS, wsIndices, nVectors, gradBuff, C, accuracyThreshold, tau, innerMaxIterations, alphaBuff,
                                            deltaalphaBuff, resinfoBuff, nWS));

        {
            auto resinfoHostPtr        = resinfoBuff.toHost(ReadWriteMode::readOnly, &status);
            auto resinfoHost           = resinfoHostPtr.get();
            size_t localInnerIteration = size_t(resinfoHost[0]);
            diff                       = resinfoHost[1];
            innerIteration += localInnerIteration;
        }

        DAAL_CHECK_STATUS(status, updateGrad(kernelWS, deltaalphaBuff, gradBuff, nVectors, nWS));

        if (checkStopCondition(diff, diffPrev, accuracyThreshold, sameLocalDiff)) break;
        diffPrev = diff;
    }

    SaveResultModel<algorithmFPType> result(alphaBuff, gradBuff, yBuff, C, nVectors);

    DAAL_CHECK_STATUS(status, result.init());
    DAAL_CHECK_STATUS(status, result.setResultsToModel(xTable, *static_cast<Model *>(r)));
    DAAL_CHECK_STATUS(status, yTable.releaseBlockOfRows(yBD));

    return status;
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
