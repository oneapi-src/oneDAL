/* file: df_regression_train_kernel_oneapi.h */
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
//  training for GPU.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_KERNEL_ONEAPI_H__
#define __DF_REGRESSION_TRAIN_KERNEL_ONEAPI_H__

#include "oneapi/internal/types.h"
#include "oneapi/internal/execution_context.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/kernel/dtrees/forest/regression/df_regression_model_impl.h"
#include "algorithms/decision_forest/decision_forest_regression_training_types.h"
#include "algorithms/decision_forest/decision_forest_regression_model.h"
#include "algorithms/kernel/dtrees/forest/oneapi/df_feature_type_helper_oneapi.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, Method method>
class RegressionTrainBatchKernelOneAPI : public daal::algorithms::Kernel
{
public:
    RegressionTrainBatchKernelOneAPI() {}
    services::Status compute(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, decision_forest::regression::Model & m,
                             Result & res, const Parameter & par);

private:
    services::Status initializeTreeOrder(size_t nRows, oneapi::internal::UniversalBuffer & treeOrder);

    services::Status convertSplitToLeaf(oneapi::internal::UniversalBuffer & nodeList, size_t nNodes);

    services::Status markPresentRows(const oneapi::internal::UniversalBuffer & rowsList, oneapi::internal::UniversalBuffer & rowsBuffer, size_t nRows,
                                     size_t localSize, size_t nSubgroupSums);
    services::Status countAbsentRowsForBlocks(const oneapi::internal::UniversalBuffer & rowsBuffer, size_t nRows,
                                              oneapi::internal::UniversalBuffer & partialSums, size_t localSize, size_t nSubgroupSums);
    services::Status countAbsentRowsTotal(const oneapi::internal::UniversalBuffer & partialSums,
                                          oneapi::internal::UniversalBuffer & partialPrefixSums, oneapi::internal::UniversalBuffer & totalSum,
                                          size_t localSize, size_t nSubgroupSums);
    services::Status fillOOBRowsListByBlocks(const oneapi::internal::UniversalBuffer & rowsBuffer, size_t nRows,
                                             const oneapi::internal::UniversalBuffer & partialPrefixSums,
                                             oneapi::internal::UniversalBuffer & oobRowsList, size_t localSize, size_t nSubgroupSums);

    services::Status getOOBRows(const oneapi::internal::UniversalBuffer & rowsList, size_t nRows, size_t & nOOBRows,
                                oneapi::internal::UniversalBuffer & oobRowsList);

    services::Status getNumOfSplitNodes(const oneapi::internal::UniversalBuffer & nodeList, size_t nNodes, size_t & nSplitNodes);

    services::Status doNodesSplit(const oneapi::internal::UniversalBuffer & nodeList, size_t nNodes, oneapi::internal::UniversalBuffer & nodeListNew);

    services::Status splitNodeListOnGroupsBySize(const oneapi::internal::UniversalBuffer & nodeList, size_t nNodes,
                                                 oneapi::internal::UniversalBuffer & bigNodesGroups, oneapi::internal::UniversalBuffer & nodeIndeces);

    services::Status doLevelPartition(const oneapi::internal::UniversalBuffer & data, oneapi::internal::UniversalBuffer & nodeList, size_t nNodes,
                                      oneapi::internal::UniversalBuffer & treeOrder, oneapi::internal::UniversalBuffer & treeOrderBuf, size_t nRows,
                                      size_t nFeatures);

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

    services::Status partitionCopy(oneapi::internal::UniversalBuffer & treeOrderBuf, oneapi::internal::UniversalBuffer & treeOrder, size_t iStart,
                                   size_t nRows);

    services::Status updateMDIVarImportance(const oneapi::internal::UniversalBuffer & nodeList,
                                            const oneapi::internal::UniversalBuffer & nodeImpDecreaseList, size_t nNodes,
                                            services::Buffer<algorithmFPType> & varImp, size_t nFeatures);

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

    oneapi::internal::KernelPtr kernelBumper;

    oneapi::internal::KernelPtr kernelInitializeTreeOrder;
    oneapi::internal::KernelPtr kernelComputePartialHistograms;
    oneapi::internal::KernelPtr kernelReducePartialHistograms;
    oneapi::internal::KernelPtr kernelComputeBestSplitByHistogram;
    oneapi::internal::KernelPtr kernelComputeBestSplitSinglePass;
    oneapi::internal::KernelPtr kernelPartitionCopy;

    oneapi::internal::KernelPtr kernelConvertSplitToLeaf;
    oneapi::internal::KernelPtr kernelGetNumOfSplitNodes;
    oneapi::internal::KernelPtr kernelDoNodesSplit;
    oneapi::internal::KernelPtr kernelDoLevelPartition;
    oneapi::internal::KernelPtr kernelSplitNodeListOnGroupsBySize;

    oneapi::internal::KernelPtr kernelMarkPresentRows;
    oneapi::internal::KernelPtr kernelCountAbsentRowsForBlocks;
    oneapi::internal::KernelPtr kernelCountAbsentRowsTotal;
    oneapi::internal::KernelPtr kernelFillOOBRowsListByBlocks;

    oneapi::internal::KernelPtr kernelUpdateMDIVarImportance;

    const uint32_t _maxWorkItemsPerGroup = 256;   // should be a power of two for interal needs
    const uint32_t _maxLocalBuffer       = 30000; // should be less than a half of local memory (two buffers)
    const uint32_t _preferableSubGroup   = 16;    // preferable maximal sub-group size
    const uint32_t _maxLocalSize         = 128;
    const uint32_t _maxLocalSums         = 256;
    const uint32_t _maxLocalHistograms   = 256;
    const uint32_t _preferableGroupSize  = 256;
    const uint32_t _minRowsBlock         = 256;
    const uint32_t _maxBins              = 256;

    const uint32_t _nHistProps     = 3; // number of properties in bins histogram (i.e. n, mean and var)
    const uint32_t _nNodesGroups   = 3; // all nodes are split on groups (big, medium, small)
    const uint32_t _nodeGroupProps = 2; // each nodes Group contains props: numOfNodes, maxNumOfBlocks
};

} // namespace internal
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
