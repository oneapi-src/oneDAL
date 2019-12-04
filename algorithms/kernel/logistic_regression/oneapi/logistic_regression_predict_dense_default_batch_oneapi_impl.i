/* file: logistic_regression_predict_dense_default_batch_oneapi_impl.i */
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

/*
//++
//  Implemetation prediction of logistic regression on the GPU
//--
*/

#ifndef __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_ONEAPI_IMPL_I__
#define __LOGISTIC_REGRESSION_PREDICT_DENSE_DEFAULT_BATCH_ONEAPI_IMPL_I__

#include "logistic_regression_model_impl.h"
#include "cl_kernel/logistic_regression_dense_default.cl"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace prediction
{
namespace internal
{
using namespace daal::oneapi::internal;

// Heaviside step function
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictBatchKernelOneAPI<algorithmFPType, method, cpu>::heaviside(const services::Buffer<algorithmFPType> & x,
                                                                                   services::Buffer<algorithmFPType> & result, const uint32_t n)
{
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_logistic_regression_prediction_");
    cachekey.add(options);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelLogisticResgression, options.c_str());

    const char * const kernelName = "heaviside";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(2);
    args.set(0, x, AccessModeIds::read);
    args.set(1, result, AccessModeIds::write);

    KernelRange range(n);

    ctx.run(range, kernel, args, &status);

    return status;
}

// Index max elements for each row
template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictBatchKernelOneAPI<algorithmFPType, method, cpu>::argMax(const services::Buffer<algorithmFPType> & x,
                                                                                services::Buffer<algorithmFPType> & result, const uint32_t n,
                                                                                const uint32_t p)
{
    services::Status status;

    ExecutionContextIface & ctx    = services::Environment::getInstance()->getDefaultExecutionContext();
    ClKernelFactoryIface & factory = ctx.getClKernelFactory();

    const services::String options = getKeyFPType<algorithmFPType>();
    services::String cachekey("__daal_algorithms_logistic_regression_prediction_");
    cachekey.add(options);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), clKernelLogisticResgression, options.c_str());

    const char * const kernelName = "argMax";
    KernelPtr kernel              = factory.getKernel(kernelName);

    KernelArguments args(3);
    args.set(0, x, AccessModeIds::read);
    args.set(1, result, AccessModeIds::write);
    args.set(2, p);

    KernelRange range(n);

    ctx.run(range, kernel, args, &status);

    return status;
}

template <typename algorithmFPType, prediction::Method method, CpuType cpu>
services::Status PredictBatchKernelOneAPI<algorithmFPType, method, cpu>::compute(services::HostAppIface * pHostApp, NumericTable * x,
                                                                                 const logistic_regression::Model * m, size_t nClasses,
                                                                                 NumericTable * pRes, NumericTable * pProb, NumericTable * pLogProb)
{
    services::Status status;

    auto & ctx = services::Environment::getInstance()->getDefaultExecutionContext();

    const daal::algorithms::logistic_regression::internal::ModelImpl * pModel =
        static_cast<const daal::algorithms::logistic_regression::internal::ModelImpl *>(m);

    const size_t n = x->getNumberOfRows();
    const size_t p = x->getNumberOfColumns();

    const bool isBinary = nClasses == 2;

    NumericTablePtr beta = pModel->getBeta();

    // X
    BlockDescriptor<algorithmFPType> xBlock;
    DAAL_CHECK_STATUS(status, x->getBlockOfRows(0, n, ReadWriteMode::readOnly, xBlock));
    const services::Buffer<algorithmFPType> xBuff = xBlock.getBuffer();

    // Beta
    DAAL_ASSERT(beta->getNumberOfRows() == nClasses);
    DAAL_ASSERT(beta->getNumberOfColumns() == p + 1);

    BlockDescriptor<algorithmFPType> betaBlock;
    DAAL_CHECK_STATUS(status, beta->getBlockOfRows(0, p, ReadWriteMode::readOnly, betaBlock));
    const services::Buffer<algorithmFPType> betaBuff = betaBlock.getBuffer();

    //compute
    DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_fUniversal, n * p));
    services::Buffer<algorithmFPType> fBuf = _fUniversal.get<algorithmFPType>();

    const uint32_t offset = uint32_t(1);

    if (isBinary)
    {
        DAAL_CHECK_STATUS(status, LogisticLoss::applyBeta(xBuff, betaBuff, fBuf, n, p, offset));
        DAAL_CHECK_STATUS(status, LogisticLoss::betaIntercept(betaBuff, fBuf, n));
    }
    else
    {
        DAAL_CHECK_STATUS(status, CrossEntropyLoss::applyBeta(xBuff, betaBuff, fBuf, n, nClasses, p, p + 1, offset));
        DAAL_CHECK_STATUS(status, HelperObjectiveFunction::lazyAllocate(_oneVector, n));
        services::Buffer<algorithmFPType> oneVectorBuf = _oneVector.get<algorithmFPType>();
        ctx.fill(_oneVector, 1.0, &status);

        DAAL_CHECK_STATUS(status, CrossEntropyLoss::betaIntercept(oneVectorBuf, betaBuff, fBuf, n, nClasses, p + 1));
    }

    if (pProb || pLogProb)
    {
        // before transforming raw values to sigmoid and logarithm, predict labels

        NumericTable * pRaw = pProb ? pProb : pLogProb;

        DAAL_ASSERT(pRaw->getNumberOfRows() == n);
        DAAL_ASSERT(pRaw->getNumberOfColumns() == nClasses);

        BlockDescriptor<algorithmFPType> rawBlock;
        DAAL_CHECK_STATUS(status, pRaw->getBlockOfRows(0, n, ReadWriteMode::writeOnly, rawBlock));
        services::Buffer<algorithmFPType> aRawBuff = rawBlock.getBuffer();

        if (isBinary)
        {
            bool calculateInverse = true;
            DAAL_CHECK_STATUS(status, LogisticLoss::sigmoids(fBuf, aRawBuff, n, calculateInverse));
        }
        else
        {
            DAAL_CHECK_STATUS(status, CrossEntropyLoss::softmax(fBuf, aRawBuff, n, nClasses));
        }

        if (pLogProb)
        {
            if (pProb)
            {
                DAAL_ASSERT(pLogProb->getNumberOfRows() == n);
                DAAL_ASSERT(pLogProb->getNumberOfColumns() == nClasses);

                BlockDescriptor<algorithmFPType> logProbBlock;
                DAAL_CHECK_STATUS(status, pLogProb->getBlockOfRows(0, n, ReadWriteMode::writeOnly, logProbBlock));
                services::Buffer<algorithmFPType> logProbBuff = logProbBlock.getBuffer();

                DAAL_CHECK_STATUS(status, math::vLog(aRawBuff, logProbBuff, n * nClasses));

                DAAL_CHECK_STATUS(status, pLogProb->releaseBlockOfRows(logProbBlock));
            }
            else
            {
                DAAL_CHECK_STATUS(status, math::vLog(aRawBuff, aRawBuff, n * nClasses));
            }
        }
        DAAL_CHECK_STATUS(status, pRaw->releaseBlockOfRows(rawBlock));
    }

    if (pRes)
    {
        DAAL_ASSERT(pRes->getNumberOfRows() == n);
        DAAL_ASSERT(pRes->getNumberOfColumns() == 1);

        BlockDescriptor<algorithmFPType> yBlock;
        DAAL_CHECK_STATUS(status, pRes->getBlockOfRows(0, n, ReadWriteMode::readWrite, yBlock));
        services::Buffer<algorithmFPType> yBuff = yBlock.getBuffer();

        if (isBinary)
        {
            DAAL_CHECK_STATUS(status, heaviside(fBuf, yBuff, n));
        }
        else
        {
            DAAL_CHECK_STATUS(status, argMax(fBuf, yBuff, n, nClasses));
        }

        DAAL_CHECK_STATUS(status, pRes->releaseBlockOfRows(yBlock));
    }

    DAAL_CHECK_STATUS(status, x->releaseBlockOfRows(xBlock));
    DAAL_CHECK_STATUS(status, beta->releaseBlockOfRows(betaBlock));

    return status;
}

} // namespace internal
} // namespace prediction
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

#endif
