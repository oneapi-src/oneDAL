/* file: df_regression_train_hist_oneapi_impl.i */
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
//  Implementation of auxiliary functions for decision forest regression
//  hist method.
//--
*/

#ifndef __DF_REGRESSION_TRAIN_HIST_ONEAPI_IMPL_I__
#define __DF_REGRESSION_TRAIN_HIST_ONEAPI_IMPL_I__

#include "algorithms/kernel/dtrees/forest/regression/oneapi/df_regression_train_hist_kernel_oneapi.h"
#include "algorithms/kernel/engines/engine_types_internal.h"
#include "algorithms/kernel/dtrees/forest/regression/oneapi/cl_kernels/df_batch_regression_kernels.cl"

#include "algorithms/kernel/dtrees/forest/oneapi/df_feature_type_helper_oneapi.i"
#include "algorithms/kernel/dtrees/forest/regression/df_regression_model_impl.h"
#include "algorithms/kernel/dtrees/forest/regression/oneapi/df_regression_tree_helper_impl.i"

#include "externals/service_ittnotify.h"
#include "externals/service_rng.h"
#include "externals/service_math.h" //will remove after migrating finalize MDA to GPU
#include "services/buffer.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "services/error_indexes.h"
#include "service/kernel/service_data_utils.h"
#include "service/kernel/service_algo_utils.h"
#include "service/kernel/service_arrays.h"
#include "service/kernel/service_utils.h"
#include "oneapi/internal/types.h"

using namespace daal::algorithms::decision_forest::internal;
using namespace daal::algorithms::decision_forest::regression::internal;

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
template <typename algorithmFPType>
static services::String getFPTypeAccuracy()
{
    if (IsSameType<algorithmFPType, float>::value)
    {
        return services::String(" -D algorithmFPTypeAccuracy=(float)1e-5 ");
    }
    if (IsSameType<algorithmFPType, double>::value)
    {
        return services::String(" -D algorithmFPTypeAccuracy=(double)1e-10 ");
    }
    return services::String();
}

template <typename algorithmFPType>
static void buildProgram(ClKernelFactoryIface & factory)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
    {
        auto fptype_name     = getKeyFPType<algorithmFPType>();
        auto fptype_accuracy = getFPTypeAccuracy<algorithmFPType>();
        auto build_options   = fptype_name;
        build_options.add(" -D NODE_PROPS=5 -D IMPURITY_PROPS=2 -D HIST_PROPS=3 -D BIG_NODE_LOW_BORDER_BLOCKS_NUM=32 -D LOCAL_BUFFER_SIZE=256 ");
        build_options.add(fptype_accuracy);
        build_options.add("-cl-std=CL1.2");

        services::String cachekey("__daal_algorithms_df_batch_regression_");
        cachekey.add(fptype_name);
        factory.build(ExecutionTargetIds::device, cachekey.c_str(), df_batch_regression_kernels, build_options.c_str());
    }
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::initializeTreeOrder(size_t nRows, UniversalBuffer & treeOrder)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initializeTreeOrder);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelInitializeTreeOrder;

    {
        KernelArguments args(1);
        args.set(0, treeOrder, AccessModeIds::write);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::markPresentRows(const UniversalBuffer & rowsList,
                                                                                          UniversalBuffer & rowsBuffer, size_t nRows,
                                                                                          size_t localSize, size_t nSubgroupSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.markPresentRows);
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    {
        auto & kernel = kernelMarkPresentRows;
        KernelArguments args(3);
        args.set(0, rowsList, AccessModeIds::read);
        args.set(1, rowsBuffer, AccessModeIds::write);
        args.set(2, (int32_t)nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nSubgroupSums);

        KernelNDRange range(1);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::countAbsentRowsForBlocks(const UniversalBuffer & rowsBuffer, size_t nRows,
                                                                                                   UniversalBuffer & partialSums, size_t localSize,
                                                                                                   size_t nSubgroupSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countAbsentRowsForBlocks);
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    {
        auto & kernel = kernelCountAbsentRowsForBlocks;
        KernelArguments args(3);
        args.set(0, rowsBuffer, AccessModeIds::read);
        args.set(1, partialSums, AccessModeIds::write);
        args.set(2, (int32_t)nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nSubgroupSums);

        KernelNDRange range(1);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::countAbsentRowsTotal(const UniversalBuffer & partialSums,
                                                                                               UniversalBuffer & partialPrefixSums,
                                                                                               UniversalBuffer & totalSum, size_t localSize,
                                                                                               size_t nSubgroupSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.countAbsentRowsTotal);
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    {
        auto & kernel = kernelCountAbsentRowsTotal;
        KernelArguments args(4);
        args.set(0, partialSums, AccessModeIds::read);
        args.set(1, partialPrefixSums, AccessModeIds::write);
        args.set(2, totalSum, AccessModeIds::write);
        args.set(3, (int32_t)nSubgroupSums);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::fillOOBRowsListByBlocks(const UniversalBuffer & rowsBuffer, size_t nRows,
                                                                                                  const UniversalBuffer & partialPrefixSums,
                                                                                                  UniversalBuffer & oobRowsList, size_t localSize,
                                                                                                  size_t nSubgroupSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.fillOOBRowsListByBlocks);
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    {
        auto & kernel = kernelFillOOBRowsListByBlocks;
        KernelArguments args(4);
        args.set(0, rowsBuffer, AccessModeIds::read);
        args.set(1, partialPrefixSums, AccessModeIds::read);
        args.set(2, oobRowsList, AccessModeIds::write);
        args.set(3, (int32_t)nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nSubgroupSums);

        KernelNDRange range(1);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}
template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::getOOBRows(const UniversalBuffer & rowsList, size_t nRows,
                                                                                     size_t & nOOBRows, UniversalBuffer & oobRowsList)
{
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    const int absentMark    = -1;
    const int localSize     = _preferableSubGroup;
    const int nSubgroupSums = _maxLocalSums * localSize < nRows ? _maxLocalSums : (nRows / localSize + !(nRows / localSize));

    auto rowsBuffer        = context.allocate(TypeIds::id<int>(), nRows, &status); // it is filled with marks Present/Absent for each rows
    auto partialSums       = context.allocate(TypeIds::id<int>(), nSubgroupSums, &status);
    auto partialPrefixSums = context.allocate(TypeIds::id<int>(), nSubgroupSums, &status);
    auto totalSum          = context.allocate(TypeIds::id<int>(), 1, &status);
    DAAL_CHECK_STATUS_VAR(status);

    context.fill(rowsBuffer, absentMark, &status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(markPresentRows(rowsList, rowsBuffer, nRows, localSize, nSubgroupSums));
    DAAL_CHECK_STATUS_VAR(countAbsentRowsForBlocks(rowsBuffer, nRows, partialSums, localSize, nSubgroupSums));
    DAAL_CHECK_STATUS_VAR(countAbsentRowsTotal(partialSums, partialPrefixSums, totalSum, localSize, nSubgroupSums));

    auto nOOBRowsHost = totalSum.template get<int>().toHost(ReadWriteMode::readOnly);
    DAAL_CHECK_MALLOC(nOOBRowsHost.get());

    nOOBRows = (size_t)nOOBRowsHost.get()[0];

    if (nOOBRows > 0)
    {
        // assign buffer of required size to the input oobRowsList buffer
        oobRowsList = context.allocate(TypeIds::id<int>(), nOOBRows, &status);
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS_VAR(fillOOBRowsListByBlocks(rowsBuffer, nRows, partialPrefixSums, oobRowsList, localSize, nSubgroupSums));
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::getNumOfSplitNodes(const UniversalBuffer & nodeList, size_t nNodes,
                                                                                             size_t & nSplitNodes)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getNumOfSplitNodes);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelGetNumOfSplitNodes;

    auto bufNSplitNodes = context.allocate(TypeIds::id<int>(), 1, &status);
    DAAL_CHECK_STATUS_VAR(status);

    {
        KernelArguments args(3);
        args.set(0, nodeList, AccessModeIds::read);
        args.set(1, (int32_t)nNodes);
        args.set(2, bufNSplitNodes, AccessModeIds::write);

        size_t localSize = _preferableSubGroup;

        // will add more range for it
        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    auto bufNsplitNodesHost = bufNSplitNodes.template get<int>().toHost(ReadWriteMode::readOnly);
    nSplitNodes             = bufNsplitNodesHost.get()[0];

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::convertSplitToLeaf(UniversalBuffer & nodeList, size_t nNodes)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.convertSplitToLeaf);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelConvertSplitToLeaf;

    {
        KernelArguments args(1);
        args.set(0, nodeList, AccessModeIds::readwrite);

        KernelRange global_range(nNodes);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::doNodesSplit(const UniversalBuffer & nodeList, size_t nNodes,
                                                                                       UniversalBuffer & nodeListNew)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.doNodesSplit);

    /*split rows for each nodes in accordance with best split info*/

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelDoNodesSplit;

    {
        KernelArguments args(3);
        args.set(0, nodeList, AccessModeIds::read);
        args.set(1, (int32_t)nNodes);
        args.set(2, nodeListNew, AccessModeIds::write);

        size_t localSize = _preferableSubGroup;

        // will add more range for it
        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::splitNodeListOnGroupsBySize(const UniversalBuffer & nodeList, size_t nNodes,
                                                                                                      UniversalBuffer & nodesGroups,
                                                                                                      UniversalBuffer & nodeIndices)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.splitNodeListOnGroupsBySize);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelSplitNodeListOnGroupsBySize;

    {
        KernelArguments args(5);
        args.set(0, nodeList, AccessModeIds::read);
        args.set(1, (int32_t)nNodes);
        args.set(2, nodesGroups, AccessModeIds::write);
        args.set(3, nodeIndices, AccessModeIds::write);
        args.set(4, _minRowsBlock);

        size_t localSize = _preferableSubGroup;

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::doLevelPartition(const UniversalBuffer & data, UniversalBuffer & nodeList,
                                                                                           size_t nNodes, UniversalBuffer & treeOrder,
                                                                                           UniversalBuffer & treeOrderBuf, size_t nRows,
                                                                                           size_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.doLevelPartition);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelDoLevelPartition;

    {
        KernelArguments args(5);
        args.set(0, data, AccessModeIds::read);
        args.set(1, nodeList, AccessModeIds::read);
        args.set(2, treeOrder, AccessModeIds::read);
        args.set(3, treeOrderBuf, AccessModeIds::write);
        args.set(4, (int32_t)nFeatures);

        size_t localSize = _preferableSubGroup;

        KernelRange local_range(localSize, 1);
        KernelRange global_range(localSize, nNodes);

        KernelNDRange range(2);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    DAAL_CHECK_STATUS_VAR(partitionCopy(treeOrderBuf, treeOrder, 0, nRows));

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computeBestSplitByHistogram(
    const UniversalBuffer & nodeHistogramList, UniversalBuffer & selectedFeatures, size_t nSelectedFeatures, UniversalBuffer & nodeList,
    UniversalBuffer & nodeIndices, size_t nodeIndicesOffset, UniversalBuffer & binOffsets, UniversalBuffer & impList,
    UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired, size_t nNodes, size_t nMaxBinsAmongFtrs, size_t minObservationsInLeafNode,
    algorithmFPType impurityThreshold)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeBestSpitByHistogramLevel);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelComputeBestSplitByHistogram;

    {
        KernelArguments args(13);
        args.set(0, nodeHistogramList, AccessModeIds::read);
        args.set(1, selectedFeatures, AccessModeIds::read);
        args.set(2, (int32_t)nSelectedFeatures);
        args.set(3, binOffsets, AccessModeIds::read);
        args.set(4, nodeList, AccessModeIds::readwrite); // nodeList will be updated with split attributes
        args.set(5, nodeIndices, AccessModeIds::read);
        args.set(6, nodeIndicesOffset);
        args.set(7, impList, AccessModeIds::write);
        args.set(8, nodeImpDecreaseList, AccessModeIds::write);
        args.set(9, (int32_t)updateImpDecreaseRequired);
        args.set(10, (int32_t)nMaxBinsAmongFtrs);
        args.set(11, (int32_t)minObservationsInLeafNode);
        args.set(12, impurityThreshold);

        const size_t numOfSubGroupsPerNode = 4; //add logic for adjusting it in accordance with nNodes
        size_t localSize                   = _preferableSubGroup * numOfSubGroupsPerNode;

        KernelRange local_range(localSize, 1);
        KernelRange global_range(localSize, nNodes);

        KernelNDRange range(2);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computeBestSplitSinglePass(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
    const services::Buffer<algorithmFPType> & response, UniversalBuffer & binOffsets, UniversalBuffer & nodeList, UniversalBuffer & nodeIndices,
    size_t nodeIndicesOffset, UniversalBuffer & impList, UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired, size_t nFeatures,
    size_t nNodes, size_t minObservationsInLeafNode, algorithmFPType impurityThreshold)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeBestSplitSinglePass);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelComputeBestSplitSinglePass;

    {
        KernelArguments args(15);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, selectedFeatures, AccessModeIds::read);
        args.set(3, (int32_t)nSelectedFeatures);
        args.set(4, response, AccessModeIds::read);
        args.set(5, binOffsets, AccessModeIds::read);
        args.set(6, nodeList, AccessModeIds::readwrite); // nodeList will be updated with split attributes
        args.set(7, nodeIndices, AccessModeIds::read);
        args.set(8, (int32_t)nodeIndicesOffset);
        args.set(9, impList, AccessModeIds::write);
        args.set(10, nodeImpDecreaseList, AccessModeIds::write);
        args.set(11, (int32_t)updateImpDecreaseRequired);
        args.set(12, (int32_t)nFeatures);
        args.set(13, (int32_t)minObservationsInLeafNode);
        args.set(14, impurityThreshold);

        const size_t numOfSubGroupsPerNode = 4; //add logic for adjusting it in accordance with nNodes
        size_t localSize                   = _preferableSubGroup * numOfSubGroupsPerNode;

        KernelRange local_range(localSize, 1);
        KernelRange global_range(localSize, nNodes);

        KernelNDRange range(2);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computeBestSplit(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
    const services::Buffer<algorithmFPType> & response, UniversalBuffer & nodeList, UniversalBuffer & binOffsets, UniversalBuffer & impList,
    UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired, size_t nFeatures, size_t nNodes, size_t minObservationsInLeafNode,
    algorithmFPType impurityThreshold)
{
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto nodesGroups = context.allocate(TypeIds::id<int>(), _nNodesGroups * _nodeGroupProps, &status);
    auto nodeIndices = context.allocate(TypeIds::id<int>(), nNodes, &status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(splitNodeListOnGroupsBySize(nodeList, nNodes, nodesGroups, nodeIndices));

    auto nodesGroupsHost = nodesGroups.template get<int>().toHost(ReadWriteMode::readOnly);

    size_t nGroupNodes    = 0;
    size_t processedNodes = 0;
    for (size_t i = 0; i < _nNodesGroups; i++, processedNodes += nGroupNodes)
    {
        nGroupNodes = nodesGroupsHost.get()[i * _nodeGroupProps + 0];
        if (0 == nGroupNodes) continue;

        size_t maxGroupBlocksNum = nodesGroupsHost.get()[i * _nodeGroupProps + 1];

        size_t groupIndicesOffset = processedNodes;

        if (maxGroupBlocksNum > 1)
        {
            size_t nPartialHistograms = maxGroupBlocksNum < _maxLocalHistograms / 2 ? maxGroupBlocksNum : _maxLocalHistograms / 2;
            //_maxLocalHistograms/2 (128) showed better performance than _maxLocalHistograms need to investigate
            size_t nMaxBinsAmongFtrs = 256; // extract it from IndexedFeatures
            int reduceLocalSize      = 16;  // add logic for its adjustment

            size_t partHistSize = nSelectedFeatures * nMaxBinsAmongFtrs * _nHistProps;

            auto partialHistograms = context.allocate(TypeIds::id<algorithmFPType>(), nGroupNodes * nPartialHistograms * partHistSize, &status);
            auto nodesHistograms   = context.allocate(TypeIds::id<algorithmFPType>(), nGroupNodes * partHistSize, &status);
            DAAL_CHECK_STATUS_VAR(status);

            DAAL_CHECK_STATUS_VAR(computePartialHistograms(data, treeOrder, selectedFeatures, nSelectedFeatures, response, nodeList, nodeIndices,
                                                           groupIndicesOffset, binOffsets, nMaxBinsAmongFtrs, nFeatures, nGroupNodes,
                                                           partialHistograms, nPartialHistograms));
            DAAL_CHECK_STATUS_VAR(reducePartialHistograms(partialHistograms, nodesHistograms, nPartialHistograms, nGroupNodes, nSelectedFeatures,
                                                          nMaxBinsAmongFtrs, reduceLocalSize));

            DAAL_CHECK_STATUS_VAR(computeBestSplitByHistogram(nodesHistograms, selectedFeatures, nSelectedFeatures, nodeList, nodeIndices,
                                                              groupIndicesOffset, binOffsets, impList, nodeImpDecreaseList, updateImpDecreaseRequired,
                                                              nGroupNodes, nMaxBinsAmongFtrs, minObservationsInLeafNode, impurityThreshold));
        }
        else
        {
            DAAL_CHECK_STATUS_VAR(computeBestSplitSinglePass(data, treeOrder, selectedFeatures, nSelectedFeatures, response, binOffsets, nodeList,
                                                             nodeIndices, groupIndicesOffset, impList, nodeImpDecreaseList, updateImpDecreaseRequired,
                                                             nFeatures, nGroupNodes, minObservationsInLeafNode, impurityThreshold));
        }
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computePartialHistograms(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
    const services::Buffer<algorithmFPType> & response, UniversalBuffer & nodeList, UniversalBuffer & nodeIndices, size_t nodeIndicesOffset,
    UniversalBuffer & binOffsets, size_t nMaxBinsAmongFtrs, size_t nFeatures, size_t nNodes, UniversalBuffer & partialHistograms,
    size_t nPartialHistograms)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialHistograms);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelComputePartialHistograms;

    {
        KernelArguments args(11);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, nodeList, AccessModeIds::read);
        args.set(3, nodeIndices, AccessModeIds::read);
        args.set(4, (int32_t)nodeIndicesOffset);
        args.set(5, selectedFeatures, AccessModeIds::read);
        args.set(6, response, AccessModeIds::read);
        args.set(7, binOffsets, AccessModeIds::read);
        args.set(8, (int32_t)nMaxBinsAmongFtrs); // max num of bins among all ftrs
        args.set(9, (int32_t)nFeatures);
        args.set(10, partialHistograms, AccessModeIds::write);

        size_t localSize = nSelectedFeatures < _maxLocalSize ? nSelectedFeatures : _maxLocalSize;

        KernelRange local_range(1, localSize, 1);
        KernelRange global_range(nPartialHistograms, localSize, nNodes);

        KernelNDRange range(3);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::reducePartialHistograms(UniversalBuffer & partialHistograms,
                                                                                                  UniversalBuffer & histograms,
                                                                                                  size_t nPartialHistograms, size_t nNodes,
                                                                                                  size_t nSelectedFeatures, size_t nMaxBinsAmongFtrs,
                                                                                                  size_t reduceLocalSize)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.reducePartialHistograms);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelReducePartialHistograms;

    {
        KernelArguments args(5);
        args.set(0, partialHistograms, AccessModeIds::read);
        args.set(1, histograms, AccessModeIds::write);
        args.set(2, (int32_t)nPartialHistograms);
        args.set(3, (int32_t)nSelectedFeatures);
        args.set(4, (int32_t)nMaxBinsAmongFtrs); // max num of bins among all ftrs

        KernelRange local_range(1, reduceLocalSize, 1);
        KernelRange global_range(nMaxBinsAmongFtrs * nSelectedFeatures, reduceLocalSize, nNodes);

        KernelNDRange range(3);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::partitionCopy(UniversalBuffer & treeOrderBuf, UniversalBuffer & treeOrder,
                                                                                        size_t iStart, size_t nRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionCopy);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelPartitionCopy;

    {
        KernelArguments args(3);
        args.set(0, treeOrderBuf, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::write);
        args.set(2, (int32_t)iStart);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::updateMDIVarImportance(const UniversalBuffer & nodeList,
                                                                                                 const UniversalBuffer & nodeImpDecreaseList,
                                                                                                 size_t nNodes,
                                                                                                 services::Buffer<algorithmFPType> & varImp,
                                                                                                 size_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateMDIVarImportance);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelUpdateMDIVarImportance;

    {
        KernelArguments args(4);
        args.set(0, nodeList, AccessModeIds::read);
        args.set(1, nodeImpDecreaseList, AccessModeIds::read);
        args.set(2, (int32_t)nNodes);
        args.set(3, varImp, AccessModeIds::write);

        int localSize = _preferableGroupSize;
        //calculating local size in way to have all subgroups for node in one group to use local buffer
        while (localSize > nNodes && localSize > _preferableSubGroup)
        {
            localSize >>= 1;
        }

        KernelRange local_range(localSize, 1);
        KernelRange global_range(localSize, nFeatures);

        KernelNDRange range(2);
        range.local(local_range, &status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, &status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <CpuType cpu>
static void shuffle(void * state, size_t n, int * dst)
{
    RNGs<int, cpu> rng;
    int idx[2];

    for (size_t i = 0; i < n; ++i)
    {
        rng.uniform(2, idx, state, 0, n);
        daal::services::internal::swap<cpu, int>(dst[idx[0]], dst[idx[1]]);
    }
}

template <CpuType cpu>
services::Status selectParallelizationTechnique(const Parameter & par, engines::internal::ParallelizationTechnique & technique)
{
    auto engineImpl = dynamic_cast<engines::internal::BatchBaseImpl *>(par.engine.get());

    engines::internal::ParallelizationTechnique techniques[] = { engines::internal::family, engines::internal::leapfrog,
                                                                 engines::internal::skipahead };

    for (auto & t : techniques)
    {
        if (engineImpl->hasSupport(t))
        {
            technique = t;
            return services::Status();
        }
    }
    return services::Status(ErrorEngineNotSupported);
}

/* following methods are related to results computation (OBB err, varImportance MDA/MDA_Scaled)*/
/* they will be migrated on GPU when prediction layer forGPU is ready*/
template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computeResults(const dtrees::internal::Tree & t, const algorithmFPType * x,
                                                                                         const algorithmFPType * y, size_t nRows, size_t nFeatures,
                                                                                         const UniversalBuffer & oobIndices, size_t nOOB,
                                                                                         UniversalBuffer & oobBuf, algorithmFPType * varImp,
                                                                                         algorithmFPType * varImpVariance, size_t nBuiltTrees,
                                                                                         const engines::EnginePtr & engine, const Parameter & par)
{
    services::Status status;
    const bool mdaRequired(par.varImportance == decision_forest::training::MDA_Raw || par.varImportance == decision_forest::training::MDA_Scaled);

    if ((par.resultsToCompute & (decision_forest::training::computeOutOfBagError | decision_forest::training::computeOutOfBagErrorPerObservation)
         || mdaRequired)
        && nOOB)
    {
        const algorithmFPType oobError = computeOOBError(t, x, y, nRows, nFeatures, oobIndices, nOOB, oobBuf, &status);

        if (mdaRequired)
        {
            TArray<int, sse2> permutation(nOOB);
            DAAL_CHECK_MALLOC(permutation.get());
            for (size_t i = 0; i < nOOB; ++i)
            {
                permutation[i] = i;
            }

            const algorithmFPType div1 = algorithmFPType(1) / algorithmFPType(nBuiltTrees);
            daal::internal::RNGs<int, sse2> rng;
            auto engineImpl = dynamic_cast<engines::internal::BatchBaseImpl *>(engine.get());

            for (size_t ftr = 0; ftr < nFeatures; ftr++)
            {
                shuffle<sse2>(engineImpl->getState(), nOOB, permutation.get());
                const algorithmFPType permOOBError =
                    computeOOBErrorPerm(t, x, y, nRows, nFeatures, oobIndices, permutation.get(), ftr, nOOB, &status);

                const algorithmFPType diff  = (permOOBError - oobError);
                const algorithmFPType delta = diff - varImp[ftr];
                varImp[ftr] += div1 * delta;
                if (varImpVariance)
                {
                    varImpVariance[ftr] += delta * (diff - varImp[ftr]);
                }
            }
        }
        DAAL_CHECK_STATUS_VAR(status);
    }
    return status;
}

template <typename algorithmFPType>
algorithmFPType RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computeOOBError(const dtrees::internal::Tree & t, const algorithmFPType * x,
                                                                                         const algorithmFPType * y, const size_t nRows,
                                                                                         const size_t nFeatures, const UniversalBuffer & indices,
                                                                                         size_t n, UniversalBuffer oobBuf, services::Status * status)
{
    typedef DFTreeConverter<algorithmFPType, sse2> DFTreeConverterType;
    typename DFTreeConverterType::TreeHelperType mTreeHelper;

    auto rowsIndHost = indices.template get<int>().toHost(ReadWriteMode::readOnly);
    auto oobBufHost  = oobBuf.template get<algorithmFPType>().toHost(ReadWriteMode::readWrite);

    //compute prediction error on each OOB row and get its mean online formulae (Welford)
    //TODO: can be threader_for() block

    algorithmFPType mean = algorithmFPType(0);
    for (size_t i = 0; i < n; i++)
    {
        int rowInd                 = rowsIndHost.get()[i];
        algorithmFPType prediction = mTreeHelper.predict(t, &x[rowInd * nFeatures]);
        oobBufHost.get()[rowInd * 2 + 0] += prediction;
        oobBufHost.get()[rowInd * 2 + 1] += algorithmFPType(1);
        algorithmFPType val = (prediction - y[rowInd]) * (prediction - y[rowInd]);
        mean += (val - mean) / algorithmFPType(i + 1);
    }

    return mean;
}

template <typename algorithmFPType>
algorithmFPType RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computeOOBErrorPerm(
    const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, const size_t nRows, const size_t nFeatures,
    const UniversalBuffer & indices, const int * indicesPerm, const size_t testFtrInd, size_t n, services::Status * status)
{
    typedef DFTreeConverter<algorithmFPType, sse2> DFTreeConverterType;
    typename DFTreeConverterType::TreeHelperType mTreeHelper;

    auto rowsIndHost = indices.template get<int>().toHost(ReadWriteMode::readOnly);
    TArray<algorithmFPType, sse2> buf(nFeatures);
    DAAL_CHECK_MALLOC(buf.get());

    algorithmFPType mean = algorithmFPType(0);
    for (size_t i = 0; i < n; i++)
    {
        int rowInd     = rowsIndHost.get()[i];
        int rowIndPerm = indicesPerm[i];
        services::internal::tmemcpy<algorithmFPType, sse2>(buf.get(), &x[rowInd * nFeatures], nFeatures);
        buf[testFtrInd]            = x[rowIndPerm * nFeatures + testFtrInd];
        algorithmFPType prediction = mTreeHelper.predict(t, buf.get());
        algorithmFPType val        = (prediction - y[rowInd]) * (prediction - y[rowInd]);
        mean += (val - mean) / algorithmFPType(i + 1);
    }

    return mean;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::finalizeOOBError(const algorithmFPType * y, const UniversalBuffer & oobBuf,
                                                                                           const size_t nRows, algorithmFPType * res,
                                                                                           algorithmFPType * resPerObs)
{
    auto oobBufHost = oobBuf.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);

    size_t nPredicted    = 0;
    algorithmFPType _res = algorithmFPType(0);

    for (size_t i = 0; i < nRows; i++)
    {
        algorithmFPType value = oobBufHost.get()[i * 2 + 0];
        algorithmFPType count = oobBufHost.get()[i * 2 + 1];

        if (algorithmFPType(0) != count)
        {
            value /= count;
            const algorithmFPType oobForObs = (value - y[i]) * (value - y[i]);
            if (resPerObs) resPerObs[i] = oobForObs;
            _res += oobForObs;
            nPredicted++;
        }
        else if (resPerObs)
            resPerObs[i] = algorithmFPType(-1); //was not in OOB set of any tree and hence not predicted
    }

    if (res) *res = _res / algorithmFPType(nPredicted);

    return services::Status();
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::finalizeVarImp(const Parameter & par, algorithmFPType * varImp,
                                                                                         algorithmFPType * varImpVariance, size_t nFeatures)
{
    if (par.varImportance == decision_forest::training::MDA_Scaled)
    {
        if (par.nTrees > 1)
        {
            const algorithmFPType div = algorithmFPType(1) / algorithmFPType(par.nTrees);
            for (size_t i = 0; i < nFeatures; i++)
            {
                varImpVariance[i] *= div;
                if (varImpVariance[i] > algorithmFPType(0)) varImp[i] /= daal::internal::Math<algorithmFPType, sse2>::sSqrt(varImpVariance[i] * div);
            }
        }
        else
        {
            for (size_t i = 0; i < nFeatures; i++)
            {
                varImp[i] = algorithmFPType(0);
            }
        }
    }
    else if (par.varImportance == decision_forest::training::MDI)
    {
        const algorithmFPType div = algorithmFPType(1) / algorithmFPType(par.nTrees);
        for (size_t i = 0; i < nFeatures; i++) varImp[i] *= div;
    }
    return services::Status();
}

///////////////////////////////////////////////////////////////////////////////////////////
/* compute method for RegressionTrainBatchKernelOneAPI */
///////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::compute(HostAppIface * pHostApp, const NumericTable * x,
                                                                                  const NumericTable * y, decision_forest::regression::Model & m,
                                                                                  Result & res, const Parameter & par)
{
    typedef DFTreeConverter<algorithmFPType, sse2> DFTreeConverterType;
    typedef TreeLevelRecord<algorithmFPType> TreeLevel;

    services::Status status;

    const size_t nRows             = x->getNumberOfRows();
    const size_t nFeatures         = x->getNumberOfColumns();
    const size_t nSelectedFeatures = par.featuresPerNode ? par.featuresPerNode : (nFeatures > 3 ? nFeatures / 3 : 1);

    const bool mdaRequired(par.varImportance == decision_forest::training::MDA_Raw || par.varImportance == decision_forest::training::MDA_Scaled);
    const bool oobRequired =
        (par.resultsToCompute & (decision_forest::training::computeOutOfBagError | decision_forest::training::computeOutOfBagErrorPerObservation)
         || mdaRequired);

    decision_forest::regression::internal::ModelImpl & mdImpl =
        *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m);
    DAAL_CHECK_MALLOC(mdImpl.resize(par.nTrees));

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    buildProgram<algorithmFPType>(kernel_factory);

    kernelInitializeTreeOrder         = kernel_factory.getKernel("initializeTreeOrder", &status);
    kernelComputeBestSplitByHistogram = kernel_factory.getKernel("computeBestSplitByHistogram", &status);
    kernelComputeBestSplitSinglePass  = kernel_factory.getKernel("computeBestSplitSinglePass", &status);
    kernelComputePartialHistograms    = kernel_factory.getKernel("computePartialHistograms", &status);
    kernelReducePartialHistograms     = kernel_factory.getKernel("reducePartialHistograms", &status);
    kernelPartitionCopy               = kernel_factory.getKernel("partitionCopy", &status);

    kernelConvertSplitToLeaf          = kernel_factory.getKernel("convertSplitToLeaf", &status);
    kernelGetNumOfSplitNodes          = kernel_factory.getKernel("getNumOfSplitNodes", &status);
    kernelDoNodesSplit                = kernel_factory.getKernel("doNodesSplit", &status);
    kernelDoLevelPartition            = kernel_factory.getKernel("doLevelPartition", &status);
    kernelSplitNodeListOnGroupsBySize = kernel_factory.getKernel("splitNodeListOnGroupsBySize", &status);

    kernelMarkPresentRows          = kernel_factory.getKernel("markPresentRows", &status);
    kernelCountAbsentRowsForBlocks = kernel_factory.getKernel("countAbsentRowsForBlocks", &status);
    kernelCountAbsentRowsTotal     = kernel_factory.getKernel("countAbsentRowsTotal", &status);
    kernelFillOOBRowsListByBlocks  = kernel_factory.getKernel("fillOOBRowsListByBlocks", &status);
    kernelUpdateMDIVarImportance   = kernel_factory.getKernel("updateMDIVarImportance", &status);

    DAAL_CHECK_STATUS_VAR(status);

    dtrees::internal::BinParams prm(_maxBins, par.minObservationsInLeafNode);
    decision_forest::internal::IndexedFeaturesOneAPI<algorithmFPType> indexedFeatures;
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK_MALLOC(featTypes.init(*x));
    DAAL_CHECK_STATUS(status, (indexedFeatures.init(*const_cast<NumericTable *>(x), &featTypes, &prm)));

    const size_t nSelectedRows = par.observationsPerTreeFraction * nRows;
    daal::services::internal::TArray<int, sse2> selectedRowsHost(nSelectedRows);
    DAAL_CHECK_MALLOC(selectedRowsHost.get());

    auto treeOrderLev    = context.allocate(TypeIds::id<int>(), nSelectedRows, &status);
    auto treeOrderLevBuf = context.allocate(TypeIds::id<int>(), nSelectedRows, &status);

    BlockDescriptor<algorithmFPType> dataBlock;
    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->getBlockOfRows(0, nRows, readOnly, dataBlock));

    /* blocks for varImp MDI calculation */
    bool mdiRequired         = (par.varImportance == decision_forest::training::MDI);
    auto nodeImpDecreaseList = context.allocate(TypeIds::id<algorithmFPType>(), 1, &status); // holder will be reallocated in loop
    BlockDescriptor<algorithmFPType> varImpBlock;
    NumericTablePtr varImpResPtr = res.get(variableImportance);

    if (mdiRequired || mdaRequired)
    {
        DAAL_CHECK_STATUS_VAR(varImpResPtr->getBlockOfRows(0, 1, writeOnly, varImpBlock));
        context.fill(varImpBlock.getBuffer(), (algorithmFPType)0, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    /* blocks for OutOfBag error calculation */
    UniversalBuffer oobBufferPerObs;
    if (oobRequired)
    {
        // oobBufferPerObs contains pair <cumulative value, count> for all out of bag observations for all trees
        oobBufferPerObs = context.allocate(TypeIds::id<algorithmFPType>(), nRows * 2, &status);
        context.fill(oobBufferPerObs, algorithmFPType(0), &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    /* blocks for MDA scaled error calculation */
    bool mdaScaledRequired = (par.varImportance == decision_forest::training::MDA_Scaled);
    daal::services::internal::TArray<algorithmFPType, sse2> varImpVariance; // for now it is calculated on host
    if (mdaScaledRequired)
    {
        varImpVariance.reset(nFeatures);
    }

    /*init engines*/
    engines::internal::ParallelizationTechnique technique = engines::internal::family;
    selectParallelizationTechnique<sse2>(par, technique);
    engines::internal::Params<sse2> params(par.nTrees);
    for (size_t i = 0; i < par.nTrees; i++)
    {
        params.nSkip[i] = i * par.nTrees * x->getNumberOfRows() * (par.featuresPerNode + 1);
    }
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par.nTrees, sizeof(engines::EnginePtr));
    daal::services::internal::TArray<engines::EnginePtr, sse2> engines(par.nTrees);
    engines::internal::EnginesCollection<sse2> enginesCollection(par.engine, technique, params, engines, &status);
    DAAL_CHECK_STATUS_VAR(status);

    if (!par.bootstrap)
    {
        DAAL_CHECK_STATUS_VAR(initializeTreeOrder(nSelectedRows, treeOrderLev));
    }

    for (size_t iter = 0; (iter < par.nTrees) && !algorithms::internal::isCancelled(status, pHostApp); ++iter)
    {
        BlockDescriptor<algorithmFPType> responseBlock;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(y)->getBlockOfRows(0, nRows, readOnly, responseBlock));

        size_t nNodes   = 1; // num of potential nodes to split on current tree level
        size_t nOOBRows = 0;

        Collection<TreeLevel> DFTreeRecords;
        Collection<UniversalBuffer> levelNodeLists;    // lists of nodes int props(rowsOffset, rows, ftrId, ftrVal ... )
        Collection<UniversalBuffer> levelNodeImpLists; // list of nodes fptype props (impurity, mean)
        UniversalBuffer oobRows;

        levelNodeLists.push_back(context.allocate(TypeIds::id<int>(), nNodes * TreeLevel::_nNodeSplitProps, &status));
        levelNodeImpLists.push_back(context.allocate(TypeIds::id<algorithmFPType>(), nNodes * TreeLevel::_nNodeImpProps, &status));
        DAAL_CHECK_STATUS_VAR(status);

        {
            auto rootNode     = levelNodeLists[0].template get<int>().toHost(ReadWriteMode::writeOnly);
            rootNode.get()[0] = 0;             // rows offset
            rootNode.get()[1] = nSelectedRows; // num of rows
        }

        auto engineImpl = dynamic_cast<engines::internal::BatchBaseImpl *>(engines[iter].get());
        if (!engineImpl) return Status(ErrorEngineNotSupported);

        if (par.bootstrap)
        {
            // TODO migrate to gpu generators and gpu sort version
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.RNG);
            daal::internal::RNGs<int, sse2> rng;
            rng.uniform(nSelectedRows, selectedRowsHost.get(), engineImpl->getState(), 0, nRows);
            daal::algorithms::internal::qSort<int, sse2>(nSelectedRows, selectedRowsHost.get());

            context.copy(treeOrderLev, 0, (void *)selectedRowsHost.get(), 0, nSelectedRows, &status);
            DAAL_CHECK_STATUS_VAR(status);
        }

        if (oobRequired)
        {
            getOOBRows(treeOrderLev, nSelectedRows, nOOBRows, oobRows); // nOOBRows and oobRows are the output
        }

        for (size_t level = 0; nNodes > 0; level++)
        {
            auto nodeList = levelNodeLists[level];
            auto impList  = levelNodeImpLists[level];

            daal::services::internal::TArray<int, sse2> selectedFeaturesHost(
                (nNodes + 1) * nSelectedFeatures); // first part is used features indices, +1 - part for generator
            DAAL_CHECK_MALLOC(selectedFeaturesHost.get());

            auto selectedFeaturesCom = context.allocate(TypeIds::id<int>(), nNodes * nSelectedFeatures, &status);
            DAAL_CHECK_STATUS_VAR(status);

            if (nSelectedFeatures != nFeatures)
            {
                daal::internal::RNGs<int, sse2> rng;
                for (size_t node = 0; node < nNodes; node++)
                {
                    rng.uniformWithoutReplacement(nSelectedFeatures, selectedFeaturesHost.get() + node * nSelectedFeatures,
                                                  selectedFeaturesHost.get() + (node + 1) * nSelectedFeatures, engineImpl->getState(), 0, nFeatures);
                }
            }
            else
            {
                for (size_t node = 0; node < nNodes; node++)
                {
                    for (size_t i = 0; i < nSelectedFeatures; i++)
                    {
                        selectedFeaturesHost.get()[node * nSelectedFeatures + i] = i;
                    }
                }
            }

            context.copy(selectedFeaturesCom, 0, (void *)selectedFeaturesHost.get(), 0, nSelectedFeatures * nNodes, &status);
            DAAL_CHECK_STATUS_VAR(status);

            if (mdiRequired)
            {
                nodeImpDecreaseList = context.allocate(TypeIds::id<algorithmFPType>(), nNodes, &status);
            }

            DAAL_CHECK_STATUS_VAR(computeBestSplit(indexedFeatures.getFullData(), treeOrderLev, selectedFeaturesCom, nSelectedFeatures,
                                                   responseBlock.getBuffer(), nodeList, indexedFeatures.binOffsets(), impList, nodeImpDecreaseList,
                                                   mdiRequired, nFeatures, nNodes, par.minObservationsInLeafNode, par.impurityThreshold));

            if (par.maxTreeDepth > 0 && par.maxTreeDepth == level)
            {
                DAAL_CHECK_STATUS_VAR(convertSplitToLeaf(nodeList, nNodes));
                DFTreeRecords.push_back(TreeLevel(nodeList, impList, nNodes));
                break;
            }

            DFTreeRecords.push_back(TreeLevel(nodeList, impList, nNodes));

            if (mdiRequired)
            {
                /*mdi is calculated only on split nodes and not calculated on last level*/
                auto varImpBuffer = varImpBlock.getBuffer();
                DAAL_CHECK_STATUS_VAR(updateMDIVarImportance(nodeList, nodeImpDecreaseList, nNodes, varImpBuffer, nFeatures));
            }

            size_t nNodesNewLevel;
            DAAL_CHECK_STATUS_VAR(getNumOfSplitNodes(nodeList, nNodes, nNodesNewLevel));

            if (nNodesNewLevel)
            {
                /*there are split nodes -> next level is required*/
                nNodesNewLevel *= 2;

                auto nodeListNewLevel = context.allocate(TypeIds::id<int>(), nNodesNewLevel * TreeLevel::_nNodeSplitProps, &status);
                auto impListNewLevel  = context.allocate(TypeIds::id<algorithmFPType>(), nNodesNewLevel * TreeLevel::_nNodeImpProps, &status);
                DAAL_CHECK_STATUS_VAR(status);

                DAAL_CHECK_STATUS_VAR(doNodesSplit(nodeList, nNodes, nodeListNewLevel));

                levelNodeLists.push_back(nodeListNewLevel);
                levelNodeImpLists.push_back(impListNewLevel);

                DAAL_CHECK_STATUS_VAR(
                    doLevelPartition(indexedFeatures.getFullData(), nodeList, nNodes, treeOrderLev, treeOrderLevBuf, nSelectedRows, nFeatures));
            }

            nNodes = nNodesNewLevel;
        } // for level

        services::Collection<SharedPtr<algorithmFPType> > binValuesHost(nFeatures);
        DAAL_CHECK_MALLOC(binValuesHost.data());
        services::Collection<algorithmFPType *> binValues(nFeatures);
        DAAL_CHECK_MALLOC(binValues.data());

        for (size_t i = 0; i < nFeatures; i++)
        {
            binValuesHost[i] = indexedFeatures.binBorders(i).template get<algorithmFPType>().toHost(ReadWriteMode::readOnly);
            binValues[i]     = binValuesHost[i].get();
        }

        typename DFTreeConverterType::TreeHelperType mTreeHelper;

        DFTreeConverterType converter;
        converter.convertToDFDecisionTree(DFTreeRecords, binValues.data(), mTreeHelper);

        mdImpl.add(mTreeHelper._tree, 0 /*nClasses*/);

        DAAL_CHECK_STATUS_VAR(computeResults(mTreeHelper._tree, dataBlock.getBlockPtr(), responseBlock.getBlockPtr(), nSelectedRows, nFeatures,
                                             oobRows, nOOBRows, oobBufferPerObs, varImpBlock.getBlockPtr(), varImpVariance.get(), iter + 1,
                                             engines[iter], par));

        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(y)->releaseBlockOfRows(responseBlock));
    }

    /* Finalize results */
    if (par.resultsToCompute & (decision_forest::training::computeOutOfBagError | decision_forest::training::computeOutOfBagErrorPerObservation))
    {
        BlockDescriptor<algorithmFPType> responseBlock;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(y)->getBlockOfRows(0, nRows, readOnly, responseBlock));

        NumericTablePtr oobErrPtr = res.get(outOfBagError);
        BlockDescriptor<algorithmFPType> oobErrBlock;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagError)
            DAAL_CHECK_STATUS_VAR(oobErrPtr->getBlockOfRows(0, 1, writeOnly, oobErrBlock));

        NumericTablePtr oobErrPerObsPtr = res.get(outOfBagErrorPerObservation);
        BlockDescriptor<algorithmFPType> oobErrPerObsBlock;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
            DAAL_CHECK_STATUS_VAR(oobErrPerObsPtr->getBlockOfRows(0, nRows, writeOnly, oobErrPerObsBlock));

        DAAL_CHECK_STATUS_VAR(
            finalizeOOBError(responseBlock.getBlockPtr(), oobBufferPerObs, nRows, oobErrBlock.getBlockPtr(), oobErrPerObsBlock.getBlockPtr()));

        if (oobErrPtr) DAAL_CHECK_STATUS_VAR(oobErrPtr->releaseBlockOfRows(oobErrBlock));

        if (oobErrPerObsPtr) DAAL_CHECK_STATUS_VAR(oobErrPerObsPtr->releaseBlockOfRows(oobErrPerObsBlock));

        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(y)->releaseBlockOfRows(responseBlock));
    }

    if (par.varImportance != decision_forest::training::none && par.varImportance != decision_forest::training::MDA_Raw)
    {
        finalizeVarImp(par, varImpBlock.getBlockPtr(), varImpVariance.get(), nFeatures);
    }

    if (mdiRequired) DAAL_CHECK_STATUS_VAR(varImpResPtr->releaseBlockOfRows(varImpBlock));

    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->releaseBlockOfRows(dataBlock));

    return status;
}

} /* namespace internal */
} /* namespace training */
} /* namespace regression */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
