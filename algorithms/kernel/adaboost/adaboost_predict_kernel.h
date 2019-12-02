/* file: adaboost_predict_kernel.h */
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
//  Declaration of template function that computes Ada Boost predictions.
//--
*/

#ifndef __ADABOOST_PREDICT_KERNEL_H__
#define __ADABOOST_PREDICT_KERNEL_H__

#include "adaboost_model.h"
#include "adaboost_predict.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_numeric_table.h"
#include "boosting_predict_kernel.h"
#include "service_environment.h"

using namespace daal::data_management;
using namespace daal::algorithms::boosting::prediction::internal;

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
template <Method method, typename algorithmFPType, CpuType cpu>
class AdaBoostPredictKernel : public Kernel
{
public:
    services::Status compute(const NumericTablePtr & x, const Model * m, const NumericTablePtr & r, const Parameter * par);
    services::Status computeTwoClassSamme(const NumericTablePtr & xTable, const Model * m, size_t nWeakLearners, const algorithmFPType * alpha,
                                          algorithmFPType * r, const Parameter * par);
    services::Status computeCommon(const NumericTablePtr & xTable, const Model * m, size_t nWeakLearners, const algorithmFPType * alpha,
                                   algorithmFPType * r, const Parameter * par);
    services::Status computeSammeProbability(algorithmFPType * p, size_t nClasses);

    services::Status processBlockSammeProbability(const size_t nRowsInCurrentBlock, algorithmFPType * p_block, const size_t nClasses,
                                                  algorithmFPType * pSumLog);

    services::Status computeClassScore(
        const size_t k, const size_t nClasses,
        daal::services::Collection<services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > > & weakPredictions,
        algorithmFPType * r, const algorithmFPType * alpha, const size_t nWeakLearners, algorithmFPType * maxClassScore);

    services::Status processBlockClassScore(
        size_t nProcessedRows, size_t nRowsInCurrentBlock, const size_t k, const size_t nClasses,
        daal::services::Collection<services::SharedPtr<daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu> > > & weakPredictions,
        algorithmFPType * curClassScore, algorithmFPType * maxClassScore, algorithmFPType * r, const algorithmFPType * alpha,
        const size_t nWeakLearners);

protected:
    size_t _nBlocks;
    size_t _nRowsInBlock;
    size_t _nRowsInLastBlock;
};

template <typename algorithmFPType, CpuType cpu>
struct TileDimensions
{
    size_t nRowsTotal       = 0;
    size_t nCols            = 0;
    size_t nRowsInBlock     = 0;
    size_t nRowsInLastBlock = 0;
    size_t nDataBlocks      = 0;

    TileDimensions(const NumericTablePtr & data, size_t nYPerRow = 1) : nRowsTotal(data->getNumberOfRows()), nCols(data->getNumberOfColumns())
    {
        nRowsInBlock     = services::internal::getNumElementsFitInMemory(services::internal::getL1CacheSize() * 0.8,
                                                                     (nCols + nYPerRow) * sizeof(algorithmFPType), nRowsInBlockDefault);
        nDataBlocks      = nRowsTotal / nRowsInBlock + !!(nRowsTotal % nRowsInBlock);
        nRowsInLastBlock = nRowsTotal - (nDataBlocks - 1) * nRowsInBlock;
    }
    static const size_t nRowsInBlockDefault = 500;
};
} // namespace internal
} // namespace prediction
} // namespace adaboost
} // namespace algorithms
} // namespace daal

#endif
