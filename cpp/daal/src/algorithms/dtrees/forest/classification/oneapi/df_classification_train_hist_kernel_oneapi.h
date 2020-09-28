/* file: df_classification_train_hist_kernel_oneapi.h */
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
//  Declaration of structure containing kernels for decision forest
//  training for GPU for the hist method.
//--
*/

#ifndef __DF_CLASSIFICATION_TRAIN_HIST_KERNEL_ONEAPI_H__
#define __DF_CLASSIFICATION_TRAIN_HIST_KERNEL_ONEAPI_H__

#include "sycl/internal/types.h"
#include "sycl/internal/execution_context.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"
#include "algorithms/decision_forest/decision_forest_classification_training_types.h"
#include "algorithms/decision_forest/decision_forest_classification_model.h"
#include "src/algorithms/dtrees/forest/oneapi/df_feature_type_helper_oneapi.h"
#include "src/algorithms/dtrees/forest/oneapi/df_tree_level_build_helper_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace classification
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method>
class ClassificationTrainBatchKernelOneAPI : public daal::algorithms::Kernel
{
public:
    ClassificationTrainBatchKernelOneAPI() {}
    services::Status compute(services::HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y,
                             decision_forest::classification::Model & m, Result & res, const Parameter & par)
    {
        return services::ErrorMethodNotImplemented;
    }

    services::Status compute(services::HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y,
                             decision_forest::classification::Model & m, Result & res,
                             const decision_forest::classification::training::interface1::Parameter & par)
    {
        return services::ErrorMethodNotImplemented;
    }
};

template <typename algorithmFPType>
class ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist> : public daal::algorithms::Kernel
{
public:
    ClassificationTrainBatchKernelOneAPI() {}
    services::Status compute(services::HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y,
                             decision_forest::classification::Model & m, Result & res,
                             const decision_forest::classification::training::interface1::Parameter & par)
    {
        Parameter tmpPar(par.nClasses);
        tmpPar.nTrees                      = par.nTrees;
        tmpPar.observationsPerTreeFraction = par.observationsPerTreeFraction;
        tmpPar.featuresPerNode             = par.featuresPerNode;
        tmpPar.maxTreeDepth                = par.maxTreeDepth;
        tmpPar.minObservationsInLeafNode   = par.minObservationsInLeafNode;
        tmpPar.seed                        = par.seed;
        tmpPar.engine                      = par.engine;
        tmpPar.impurityThreshold           = par.impurityThreshold;
        tmpPar.varImportance               = par.varImportance;
        tmpPar.resultsToCompute            = par.resultsToCompute;
        tmpPar.memorySavingMode            = par.memorySavingMode;
        tmpPar.bootstrap                   = par.bootstrap;
        return compute(pHostApp, x, y, m, res, tmpPar);
    }
    services::Status compute(services::HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y,
                             decision_forest::classification::Model & m, Result & res, const Parameter & par);

private:
    services::Status buildProgram(oneapi::internal::ClKernelFactoryIface & factory, const char * programName, const char * programSrc,
                                  const char * buildOptions);

    services::Status computeBestSplit(const oneapi::internal::UniversalBuffer & data, oneapi::internal::UniversalBuffer & treeOrder,
                                      oneapi::internal::UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
                                      const services::Buffer<algorithmFPType> & response, oneapi::internal::UniversalBuffer & nodeOffsets,
                                      oneapi::internal::UniversalBuffer & binOffsets, oneapi::internal::UniversalBuffer & splitInfo,
                                      oneapi::internal::UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired, size_t nFeatures,
                                      size_t nNodes, size_t minObservationsInLeafNode, algorithmFPType impurityThreshold);

    services::Status computeBestSplitSinglePass(const oneapi::internal::UniversalBuffer & data, oneapi::internal::UniversalBuffer & treeOrder,
                                                oneapi::internal::UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
                                                const services::Buffer<algorithmFPType> & response, oneapi::internal::UniversalBuffer & binOffsets,
                                                oneapi::internal::UniversalBuffer & nodeList, oneapi::internal::UniversalBuffer & nodeIndices,
                                                size_t nodeIndicesOffset, oneapi::internal::UniversalBuffer & impList,
                                                oneapi::internal::UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired,
                                                size_t nFeatures, size_t nNodes, size_t minObservationsInLeafNode, algorithmFPType impurityThreshold);

    services::Status computeBestSplitByHistogram(const oneapi::internal::UniversalBuffer & nodeHistogramList,
                                                 oneapi::internal::UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
                                                 oneapi::internal::UniversalBuffer & nodeList, oneapi::internal::UniversalBuffer & nodeIndices,
                                                 size_t nodeIndicesOffset, oneapi::internal::UniversalBuffer & binOffsets,
                                                 oneapi::internal::UniversalBuffer & splitInfo,
                                                 oneapi::internal::UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired,
                                                 size_t nNodes, size_t nMaxBinsAmongFtrs, size_t minObservationsInLeafNode,
                                                 algorithmFPType impurityThreshold);

    services::Status computePartialHistograms(const oneapi::internal::UniversalBuffer & data, oneapi::internal::UniversalBuffer & treeOrder,
                                              oneapi::internal::UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
                                              const services::Buffer<algorithmFPType> & response, oneapi::internal::UniversalBuffer & nodeList,
                                              oneapi::internal::UniversalBuffer & nodeIndices, size_t nodeIndicesOffset,
                                              oneapi::internal::UniversalBuffer & binOffsets, size_t nMaxBinsAmongFtrs, size_t nFeatures,
                                              size_t nNodes, oneapi::internal::UniversalBuffer & partialHistograms, size_t nPartialHistograms);

    services::Status reducePartialHistograms(oneapi::internal::UniversalBuffer & partialHistograms, oneapi::internal::UniversalBuffer & histograms,
                                             size_t nPartialHistograms, size_t nNodes, size_t nSelectedFeatures, size_t nMaxBinsAmongFtrs,
                                             size_t reduceLocalSize);

    services::Status computeResults(const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, const size_t nRows,
                                    const size_t nFeatures, const oneapi::internal::UniversalBuffer & oobIndices, size_t nOOB,
                                    oneapi::internal::UniversalBuffer & oobBuf, algorithmFPType * varImp, algorithmFPType * varImpVariance,
                                    size_t nBuiltTrees, const engines::EnginePtr & engine, const Parameter & par);

    algorithmFPType computeOOBError(const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, const size_t nRows,
                                    const size_t nFeatures, const oneapi::internal::UniversalBuffer & indices, size_t n,
                                    oneapi::internal::UniversalBuffer oobBuf, services::Status * status);

    algorithmFPType computeOOBErrorPerm(const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, const size_t nRows,
                                        const size_t nFeatures, const oneapi::internal::UniversalBuffer & indices, const int * indicesPerm,
                                        const size_t testFtrInd, size_t n, services::Status * status);

    services::Status finalizeOOBError(const algorithmFPType * y, const oneapi::internal::UniversalBuffer & oobBuf, const size_t nRows,
                                      algorithmFPType * res, algorithmFPType * resPerObs);

    services::Status finalizeVarImp(const Parameter & par, algorithmFPType * varImp, algorithmFPType * varImpVariance, size_t nFeatures);

    oneapi::internal::KernelPtr kernelComputePartialHistograms;
    oneapi::internal::KernelPtr kernelReducePartialHistograms;
    oneapi::internal::KernelPtr kernelComputeBestSplitByHistogram;
    oneapi::internal::KernelPtr kernelComputeBestSplitSinglePass;

    decision_forest::internal::TreeLevelBuildHelperOneAPI<algorithmFPType> _treeLevelBuildHelper;

    const size_t _maxWorkItemsPerGroup = 256; // should be a power of two for interal needs
    const size_t _preferableSubGroup   = 16;  // preferable maximal sub-group size
    const size_t _maxLocalSize         = 128;
    const size_t _maxLocalSums         = 256;
    const size_t _maxLocalHistograms   = 256;
    const size_t _preferableGroupSize  = 256;
    const size_t _minRowsBlock         = 256;
    const size_t _maxBins              = 256;

    const size_t _nNodesGroups   = 3; // all nodes are split on groups (big, medium, small)
    const size_t _nodeGroupProps = 2; // each nodes Group contains props: numOfNodes, maxNumOfBlocks

    static constexpr size_t _int32max = static_cast<size_t>(services::internal::MaxVal<int32_t>::get());

    size_t _nClasses;
    size_t _nMaxBinsAmongFtrs;
    size_t _totalBins;
};

} // namespace internal
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
