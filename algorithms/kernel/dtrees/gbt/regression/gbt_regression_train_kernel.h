/* file: gbt_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for gradient boosted trees
//  training.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_KERNEL_H__
#define __GBT_REGRESSION_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "gbt_regression_training_types.h"
#include "engine_batch_impl.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionTrainBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, gbt::regression::Model & m, Result & res,
                             const Parameter & par, engines::internal::BatchBaseImpl & engine);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionTrainDistrStep1Kernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable *ntBinnedData,
                             const NumericTable *ntDependentVariable,
                             const NumericTable *ntInputResponse,
                                   NumericTable *ntInputTreeStructure,
                             const NumericTable *ntInputTreeOrder,
                                   NumericTable *ntResponse,
                                   NumericTable *ntOptCoeffs,
                                   NumericTable *ntTreeOrder,
                                   NumericTable *ntFinalizedTree,
                                   NumericTable *ntTreeStructure,
                             const Parameter    &par);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionTrainDistrStep2Kernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(      NumericTable *ntInputTreeStructure,
                                   NumericTable *ntFinishedFlag);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionTrainDistrStep3Kernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable   *ntBinnedData,
                             const NumericTable   *ntBinSizes,
                                   NumericTable   *ntInputTreeStructure,
                             const NumericTable   *ntInputTreeOrder,
                             const NumericTable   *ntOptCoeffs,
                             const DataCollection *dcParentHistograms,
                                   DataCollection *dcHistograms);

private:
    template<typename BinIndexType>
    services::Status computeImpl(const BinIndexType * const data,
                                 const NumericTable        *ntBinSizes,
                                       NumericTable        *ntInputTreeStructure,
                                 const NumericTable        *ntInputTreeOrder,
                                 const NumericTable        *ntOptCoeffs,
                                 const DataCollection      *dcParentHistograms,
                                       DataCollection      *dcHistograms);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionTrainDistrStep4Kernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(      NumericTable   *ntInputTreeStructure,
                             const DataCollection *dcParentTotalHistograms,
                             const DataCollection *dcPartialHistograms,
                             const DataCollection *dcFeatureIndices,
                                   DataCollection *dcTotalHistograms,
                                   DataCollection *dcBestSplits,
                             const Parameter      &par);

private:
    services::Status packSplitIntoTable(const DAAL_INT idxFeatureBestSplit,
                                        const DAAL_INT featureIndex,
                                        const algorithmFPType impurityDecrease,
                                        const algorithmFPType leftGTotal,
                                        const algorithmFPType leftHTotal,
                                        const size_t          leftNTotal,
                                        const algorithmFPType rightGTotal,
                                        const algorithmFPType rightHTotal,
                                        const size_t          rightNTotal,
                                        NumericTablePtr &ntBestSplit);

};

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionTrainDistrStep5Kernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable   *ntBinnedData,
                             const NumericTable   *ntTransposedBinnedData,
                             const NumericTable   *ntBinSizes,
                                   NumericTable   *ntInputTreeStructure,
                             const NumericTable   *ntInputTreeOrder,
                             const DataCollection *dcPartialBestSplits,
                                   NumericTable   *ntTreeStructure,
                                   NumericTable   *ntTreeOrder,
                             const Parameter      &par);

private:
    template<typename BinIndexType>
    services::Status computeImpl(const BinIndexType   *binnedData,
                                 const BinIndexType   *transposedBinnedData,
                                 const NumericTable   *ntBinSizes,
                                       NumericTable   *ntInputTreeStructure,
                                 const NumericTable   *ntInputTreeOrder,
                                 const DataCollection *dcPartialBestSplits,
                                       NumericTable   *ntTreeStructure,
                                       NumericTable   *ntTreeOrder,
                                 const Parameter      &par);

    services::Status unpackTableIntoSplit(DAAL_INT &idxFeatureBestSplit,
                                          DAAL_INT &featureIndex,
                                          algorithmFPType &impurityDecrease,
                                          algorithmFPType &leftGTotal,
                                          algorithmFPType &leftHTotal,
                                          size_t          &leftNTotal,
                                          algorithmFPType &rightGTotal,
                                          algorithmFPType &rightHTotal,
                                          size_t          &rightNTotal,
                                          const NumericTable *ntBestSplit);

};

template <typename algorithmFPType, Method method, CpuType cpu>
class RegressionTrainDistrStep6Kernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const NumericTable           *ntInitialResponse,
                             const DataCollection         *dcBinValues,
                             const DataCollection         *dcFinalizedTrees,
                                   gbt::regression::Model *model,
                             const Parameter              &par);
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
