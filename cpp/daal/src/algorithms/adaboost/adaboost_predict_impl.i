/* file: adaboost_predict_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of Fast method for Ada Boost prediction algorithm.
//--
*/

#ifndef __ADABOOST_PREDICT_IMPL_I__
#define __ADABOOST_PREDICT_IMPL_I__

#include "src/data_management/service_numeric_table.h"
#include "services/collection.h"
#include "src/externals/service_math.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services::internal;

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostPredictKernel<method, algorithmFPType, cpu>::computeTwoClassSamme(const NumericTablePtr & xTable, const Model * boostModel,
                                                                                           size_t nWeakLearners, const algorithmFPType * alpha,
                                                                                           algorithmFPType * r, const Parameter * parameter)
{
    const size_t nVectors = xTable->getNumberOfRows();

    services::Status s;
    services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > rWeakTable =
        daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(1, nVectors, &s);
    DAAL_CHECK_STATUS_VAR(s);
    const algorithmFPType * rWeak = rWeakTable->getArray();

    services::SharedPtr<classifier::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction->clone();
    learnerPredict->enableChecks(false);
    classifier::prediction::Input * learnerInput = learnerPredict->getInput();
    DAAL_CHECK(learnerInput, services::ErrorNullInput);
    learnerInput->set(classifier::prediction::data, xTable);

    classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
    DAAL_CHECK_MALLOC(predictionRes.get())
    predictionRes->set(classifier::prediction::prediction, rWeakTable);
    DAAL_CHECK_STATUS(s, learnerPredict->setResult(predictionRes));

    const algorithmFPType zero = (algorithmFPType)0.0;
    service_memset<algorithmFPType, cpu>(r, zero, nVectors);

    const algorithmFPType one = (algorithmFPType)1.0;
    for (size_t i = 0; i < nWeakLearners; i++)
    {
        /* Get  weak learner's classification results */
        classifier::ModelPtr learnerModel = boostModel->getWeakLearnerModel(i);

        learnerInput->set(classifier::prediction::model, learnerModel);
        DAAL_CHECK_STATUS(s, learnerPredict->computeNoThrow());

        /* Update boosting classification results */
        for (size_t j = 0; j < nVectors; j++)
        {
            const algorithmFPType p = ((rWeak[j] > zero) ? one : -one);
            r[j] += p * alpha[i];
        }
    }

    for (size_t j = 0; j < nVectors; j++)
    {
        r[j] = ((r[j] >= zero) ? one : -one);
    }

    return s;
}

template <typename algorithmFPType, CpuType cpu>
struct Task
{
    DAAL_NEW_DELETE()

    Task(size_t blockSizeDefault) : bufferArray(blockSizeDefault) { buffer = bufferArray.get(); }

    TArrayCalloc<algorithmFPType, cpu> bufferArray;
    algorithmFPType * buffer;
};

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostPredictKernel<method, algorithmFPType, cpu>::computeSammeProbability(algorithmFPType * p, const size_t nClasses)
{
    daal::tls<Task<algorithmFPType, cpu> *> threadBuffer(
        [=]() -> Task<algorithmFPType, cpu> * { return new Task<algorithmFPType, cpu>(_nRowsInBlock); });

    SafeStatus safeStat;
    daal::threader_for(_nBlocks, _nBlocks, [=, &safeStat, &threadBuffer](int block) {
        const size_t nRowsToProcess = ((block == _nBlocks - 1) ? _nRowsInLastBlock : _nRowsInBlock);

        Task<algorithmFPType, cpu> * tPtr = threadBuffer.local();
        DAAL_CHECK_THR(tPtr && tPtr->buffer, ErrorMemoryAllocationFailed)

        safeStat |= processBlockSammeProbability(nRowsToProcess, &p[block * _nRowsInBlock * nClasses], nClasses, tPtr->buffer);
    });
    threadBuffer.reduce([=](Task<algorithmFPType, cpu> * v) -> void { delete (v); });
    return safeStat.detach();
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostPredictKernel<method, algorithmFPType, cpu>::processBlockSammeProbability(const size_t nRowsInCurrentBlock,
                                                                                                   algorithmFPType * p_block, const size_t nClasses,
                                                                                                   algorithmFPType * pSumLog)
{
    const algorithmFPType eps = services::internal::EpsilonVal<algorithmFPType>::get();
    algorithmFPType * pLog    = p_block;
    algorithmFPType * h_block = p_block;
    for (size_t i = 0; i < nRowsInCurrentBlock * nClasses; i++)
    {
        if (p_block[i] < eps)
        {
            pLog[i] = eps;
        }
        else
        {
            pLog[i] = p_block[i];
        }
    }

    MathInst<algorithmFPType, cpu>::vLog(nRowsInCurrentBlock * nClasses, p_block, pLog); // inplace

    service_memset<algorithmFPType, cpu>(pSumLog, 0.0, nRowsInCurrentBlock);

    const algorithmFPType nClassesMinusOne = nClasses - 1.0;
    for (size_t i = 0; i < nRowsInCurrentBlock; i++)
    {
        for (size_t j = 0; j < nClasses; j++)
        {
            pSumLog[i] += pLog[i * nClasses + j];
        }
        pSumLog[i] /= nClasses;
        for (size_t j = 0; j < nClasses; j++)
        {
            h_block[i * nClasses + j] = nClassesMinusOne * (pLog[i * nClasses + j] - pSumLog[i]); // inplace
        }
    }
    return services::Status();
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostPredictKernel<method, algorithmFPType, cpu>::computeCommon(const NumericTablePtr & xTable, const Model * boostModel,
                                                                                    size_t nWeakLearners, const algorithmFPType * alpha,
                                                                                    algorithmFPType * r, const Parameter * parameter)
{
    const size_t nClasses = parameter->nClasses;
    typedef daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> HomoNTCPU;
    const size_t nVectors = xTable->getNumberOfRows();

    daal::services::Collection<services::SharedPtr<HomoNTCPU> > weakPredictions;

    services::Status s;

    services::SharedPtr<classifier::prediction::Batch> learnerPredict = parameter->weakLearnerPrediction->clone();
    learnerPredict->enableChecks(false);
    classifier::prediction::Input * learnerInput = learnerPredict->getInput();
    DAAL_CHECK(learnerInput, services::ErrorNullInput);
    learnerInput->set(classifier::prediction::data, xTable);

    classifier::prediction::ResultPtr predictionRes(new classifier::prediction::Result());
    DAAL_CHECK_MALLOC(predictionRes.get())
    DAAL_CHECK_STATUS(s, learnerPredict->setResult(predictionRes));

    const size_t nCols                        = (method == samme) ? 1 : nClasses;
    classifier::prediction::ResultId resultId = (method == samme) ? classifier::prediction::prediction : classifier::prediction::probabilities;

    DAAL_UINT64 resToEvaluate = (method == samme) ? classifier::computeClassLabels : classifier::computeClassProbabilities;

    learnerPredict->parameter().resultsToEvaluate = resToEvaluate;

    for (size_t i = 0; i < nWeakLearners; i++)
    {
        services::SharedPtr<HomoNTCPU> rWeakTable = HomoNTCPU::create(nCols, nVectors, &s);
        weakPredictions << rWeakTable;
        DAAL_CHECK_STATUS_VAR(s);
        learnerPredict->getResult()->set(resultId, rWeakTable);
        learnerInput->set(classifier::prediction::model, boostModel->getWeakLearnerModel(i));
        DAAL_CHECK_STATUS(s, learnerPredict->computeNoThrow());
    }

    service_memset<algorithmFPType, cpu>(r, 0.0, nVectors);

    if (method == sammeR)
    {
        for (size_t m = 0; m < nWeakLearners; m++)
        {
            algorithmFPType * rWeak = weakPredictions[m]->getArray();
            computeSammeProbability(rWeak, nClasses);
        }
    }

    services::SharedPtr<HomoNTCPU> maxClassScoreTable = HomoNTCPU::create(1, nVectors, &s);
    algorithmFPType * maxClassScore                   = maxClassScoreTable->getArray();
    DAAL_CHECK(maxClassScore, services::ErrorMemoryAllocationFailed);
    service_memset<algorithmFPType, cpu>(maxClassScore, 0.0, nVectors);

    for (size_t k = 0; k < nClasses; k++)
    {
        DAAL_CHECK_STATUS(s, computeClassScore(k, nClasses, weakPredictions, r, alpha, nWeakLearners, maxClassScore));
    }
    if (nClasses == 2)
    {
        daal::threader_for(_nBlocks, _nBlocks, [=](int block) {
            const size_t nRowsToProcess = ((block == _nBlocks - 1) ? _nRowsInLastBlock : _nRowsInBlock);

            algorithmFPType * r_block      = &r[block * _nRowsInBlock];
            const algorithmFPType minusOne = -1.0;
            const algorithmFPType zero     = 0.0;
            for (size_t i = 0; i < nRowsToProcess; i++)
            {
                if (r_block[i] == zero)
                {
                    r_block[i] = minusOne;
                }
            }
        });
    }
    return s;
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostPredictKernel<method, algorithmFPType, cpu>::computeClassScore(
    const size_t k, const size_t nClasses,
    daal::services::Collection<services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > > & weakPredictions,
    algorithmFPType * r, const algorithmFPType * alpha, const size_t nWeakLearners, algorithmFPType * maxClassScore)
{
    SafeStatus safeStat;
    daal::tls<Task<algorithmFPType, cpu> *> threadBuffer(
        [=]() -> Task<algorithmFPType, cpu> * { return new Task<algorithmFPType, cpu>(_nRowsInBlock); });
    daal::threader_for(_nBlocks, _nBlocks, [=, &weakPredictions, &safeStat, &threadBuffer](int block) {
        const size_t nRowsToProcess = ((block == _nBlocks - 1) ? _nRowsInLastBlock : _nRowsInBlock);

        Task<algorithmFPType, cpu> * tPtr = threadBuffer.local();
        DAAL_CHECK_THR(tPtr && tPtr->buffer, ErrorMemoryAllocationFailed)

        safeStat |= processBlockClassScore(block * _nRowsInBlock, nRowsToProcess, k, nClasses, weakPredictions, tPtr->buffer,
                                           &maxClassScore[block * _nRowsInBlock], &r[block * _nRowsInBlock], alpha, nWeakLearners);
    });
    threadBuffer.reduce([=](Task<algorithmFPType, cpu> * v) -> void { delete (v); });
    return safeStat.detach();
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostPredictKernel<method, algorithmFPType, cpu>::processBlockClassScore(
    size_t nProcessedRows, size_t nRowsInCurrentBlock, const size_t k, const size_t nClasses,
    daal::services::Collection<services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > > & weakPredictions,
    algorithmFPType * curClassScore, algorithmFPType * maxClassScore_block, algorithmFPType * r_block, const algorithmFPType * alpha,
    const size_t nWeakLearners)
{
    const algorithmFPType k_fptype = (algorithmFPType)k;
    service_memset_seq<algorithmFPType, cpu>(curClassScore, 0.0, nRowsInCurrentBlock);
    const size_t rWeakCols = (method == samme) ? 1 : nClasses;
    for (size_t m = 0; m < nWeakLearners; m++)
    {
        const algorithmFPType * rWeak = &(weakPredictions[m]->getArray()[nProcessedRows * rWeakCols]);
        for (size_t i = 0; i < nRowsInCurrentBlock; i++)
        {
            if (method == samme)
            {
                curClassScore[i] += alpha[m] * (rWeak[i] == k_fptype);
            }
            else if (method == sammeR)
            {
                curClassScore[i] += rWeak[i * nClasses + k];
            }
        }
    }
    for (size_t i = 0; i < nRowsInCurrentBlock; i++)
    {
        if (curClassScore[i] > maxClassScore_block[i])
        {
            r_block[i]             = k;
            maxClassScore_block[i] = curClassScore[i];
        }
    }
    return services::Status();
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status AdaBoostPredictKernel<method, algorithmFPType, cpu>::compute(const NumericTablePtr & xTable, const Model * boostModel,
                                                                              const NumericTablePtr & rTable, const Parameter * par)
{
    TileDimensions<algorithmFPType, cpu> dim(xTable);
    _nRowsInBlock     = dim.nRowsInBlock;
    _nRowsInLastBlock = dim.nRowsInLastBlock;
    _nBlocks          = dim.nDataBlocks;

    const size_t nVectors = xTable->getNumberOfRows();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors, sizeof(algorithmFPType));

    const size_t nWeakLearners = boostModel->getNumberOfWeakLearners();
    services::Status s;
    WriteOnlyColumns<algorithmFPType, cpu> mtR(*rTable, 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * r = mtR.get();
    DAAL_ASSERT(r);

    const size_t nClasses = par->nClasses;
    {
        ReadColumns<algorithmFPType, cpu> mtAlpha(*boostModel->getAlpha(), 0, 0, nWeakLearners);
        DAAL_CHECK_BLOCK_STATUS(mtAlpha);
        DAAL_ASSERT(mtAlpha.get());
        if (method == samme && nClasses == 2)
        {
            DAAL_CHECK_STATUS(s, this->computeTwoClassSamme(xTable, boostModel, nWeakLearners, mtAlpha.get(), r, par));
        }
        else
        {
            DAAL_CHECK_STATUS(s, this->computeCommon(xTable, boostModel, nWeakLearners, mtAlpha.get(), r, par));
        }
    }

    return s;
}

} // namespace internal
} // namespace prediction
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
