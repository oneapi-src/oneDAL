/* file: df_regression_train_hist_oneapi_impl.i */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "src/algorithms/dtrees/forest/regression/oneapi/df_regression_train_hist_kernel_oneapi.h"
#include "src/algorithms/dtrees/forest/regression/oneapi/cl_kernels/df_batch_regression_kernels.cl"

#include "src/algorithms/dtrees/forest/oneapi/df_feature_type_helper_oneapi.i"
#include "src/algorithms/dtrees/forest/oneapi/df_tree_level_build_helper_oneapi.i"
#include "src/algorithms/dtrees/forest/regression/df_regression_model_impl.h"
#include "src/algorithms/dtrees/forest/regression/oneapi/df_regression_tree_helper_impl.i"

#include "src/externals/service_ittnotify.h"
#include "src/externals/service_rng.h"
#include "src/externals/service_math.h" //will remove after migrating finalize MDA to GPU
#include "services/internal/buffer.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "services/error_indexes.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_algo_utils.h"
#include "src/services/service_arrays.h"
#include "src/services/service_utils.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/engines/engine_types_internal.h"
#include "services/internal/sycl/types.h"

using namespace daal::algorithms::decision_forest::internal;
using namespace daal::algorithms::decision_forest::regression::internal;
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

static services::String getBuildOptions()
{
    return " -D NODE_PROPS=5 -D IMPURITY_PROPS=2 -D HIST_PROPS=3 ";
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::buildProgram(ClKernelFactoryIface & factory, const char * programName,
                                                                                       const char * programSrc, const char * buildOptions)
{
    services::Status status;

    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
    {
        auto fptype_name     = getKeyFPType<algorithmFPType>();
        auto fptype_accuracy = getFPTypeAccuracy<algorithmFPType>();
        auto build_options   = fptype_name;
        build_options.add(fptype_accuracy);
        build_options.add(" -cl-std=CL1.2 ");
        build_options.add(" -D LOCAL_BUFFER_SIZE=256 -D MAX_WORK_ITEMS_PER_GROUP=256 ");

        if (buildOptions)
        {
            build_options.add(buildOptions);
        }

        services::String cachekey("__daal_algorithms_df_batch_regression_");
        cachekey.add(build_options);
        cachekey.add(programName);

        factory.build(ExecutionTargetIds::device, cachekey.c_str(), programSrc, build_options.c_str(), status);
        DAAL_CHECK_STATUS_VAR(status);
    }

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

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelComputeBestSplitByHistogram;

    {
        DAAL_ASSERT(nSelectedFeatures <= _int32max);
        DAAL_ASSERT(nodeIndicesOffset <= _int32max);
        DAAL_ASSERT(updateImpDecreaseRequired <= _int32max);
        DAAL_ASSERT(nMaxBinsAmongFtrs <= _int32max);
        DAAL_ASSERT(minObservationsInLeafNode <= _int32max);

        DAAL_ASSERT_UNIVERSAL_BUFFER(nodeHistogramList, algorithmFPType, nNodes * nSelectedFeatures * _nMaxBinsAmongFtrs * _nHistProps);
        DAAL_ASSERT_UNIVERSAL_BUFFER(selectedFeatures, int32_t, nNodes * nSelectedFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(nodeList, int32_t, nNodes * TreeLevelRecord<algorithmFPType>::_nNodeSplitProps);
        DAAL_ASSERT_UNIVERSAL_BUFFER(nodeIndices, int32_t, nNodes);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, _nFeatures + 1);
        DAAL_ASSERT_UNIVERSAL_BUFFER(impList, algorithmFPType, nNodes * TreeLevelRecord<algorithmFPType>::_nNodeImpProps);
        if (updateImpDecreaseRequired) DAAL_ASSERT_UNIVERSAL_BUFFER(nodeImpDecreaseList, algorithmFPType, nNodes);

        KernelArguments args(13, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, nodeHistogramList, AccessModeIds::read);
        args.set(1, selectedFeatures, AccessModeIds::read);
        args.set(2, static_cast<int32_t>(nSelectedFeatures));
        args.set(3, binOffsets, AccessModeIds::read);
        args.set(4, nodeList, AccessModeIds::readwrite); // nodeList will be updated with split attributes
        args.set(5, nodeIndices, AccessModeIds::read);
        args.set(6, static_cast<int32_t>(nodeIndicesOffset));
        args.set(7, impList, AccessModeIds::write);
        args.set(8, nodeImpDecreaseList, AccessModeIds::write);
        args.set(9, static_cast<int32_t>(updateImpDecreaseRequired));
        args.set(10, static_cast<int32_t>(nMaxBinsAmongFtrs));
        args.set(11, static_cast<int32_t>(minObservationsInLeafNode));
        args.set(12, impurityThreshold);

        const size_t numOfSubGroupsPerNode = 8; //add logic for adjusting it in accordance with nNodes
        size_t localSize                   = _preferableSubGroup * numOfSubGroupsPerNode;

        KernelRange local_range(localSize, 1);
        KernelRange global_range(localSize, nNodes);

        KernelNDRange range(2);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computeBestSplitSinglePass(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
    const services::internal::Buffer<algorithmFPType> & response, UniversalBuffer & binOffsets, UniversalBuffer & nodeList,
    UniversalBuffer & nodeIndices, size_t nodeIndicesOffset, UniversalBuffer & impList, UniversalBuffer & nodeImpDecreaseList,
    bool updateImpDecreaseRequired, size_t nFeatures, size_t nNodes, size_t minObservationsInLeafNode, algorithmFPType impurityThreshold)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeBestSplitSinglePass);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelComputeBestSplitSinglePass;

    {
        DAAL_ASSERT(nSelectedFeatures <= _int32max);
        DAAL_ASSERT(nodeIndicesOffset <= _int32max);
        DAAL_ASSERT(updateImpDecreaseRequired <= _int32max);
        DAAL_ASSERT(nFeatures <= _int32max);
        DAAL_ASSERT(minObservationsInLeafNode <= _int32max);
        DAAL_ASSERT(response.size() == _nRows);

        DAAL_ASSERT_UNIVERSAL_BUFFER(data, uint32_t, _nRows * _nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int32_t, _nSelectedRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(selectedFeatures, int32_t, nNodes * nSelectedFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, _nFeatures + 1);
        DAAL_ASSERT_UNIVERSAL_BUFFER(nodeList, int32_t, nNodes * TreeLevelRecord<algorithmFPType>::_nNodeSplitProps);
        DAAL_ASSERT_UNIVERSAL_BUFFER(nodeIndices, int32_t, nNodes);
        DAAL_ASSERT_UNIVERSAL_BUFFER(impList, algorithmFPType, nNodes * TreeLevelRecord<algorithmFPType>::_nNodeImpProps);
        if (updateImpDecreaseRequired) DAAL_ASSERT_UNIVERSAL_BUFFER(nodeImpDecreaseList, algorithmFPType, nNodes);

        KernelArguments args(15, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, selectedFeatures, AccessModeIds::read);
        args.set(3, static_cast<int32_t>(nSelectedFeatures));
        args.set(4, response, AccessModeIds::read);
        args.set(5, binOffsets, AccessModeIds::read);
        args.set(6, nodeList, AccessModeIds::readwrite); // nodeList will be updated with split attributes
        args.set(7, nodeIndices, AccessModeIds::read);
        args.set(8, static_cast<int32_t>(nodeIndicesOffset));
        args.set(9, impList, AccessModeIds::write);
        args.set(10, nodeImpDecreaseList, AccessModeIds::write);
        args.set(11, static_cast<int32_t>(updateImpDecreaseRequired));
        args.set(12, static_cast<int32_t>(nFeatures));
        args.set(13, static_cast<int32_t>(minObservationsInLeafNode));
        args.set(14, impurityThreshold);

        const size_t numOfSubGroupsPerNode = 8; //add logic for adjusting it in accordance with nNodes
        size_t localSize                   = _preferableSubGroup * numOfSubGroupsPerNode;

        KernelRange local_range(localSize, 1);
        KernelRange global_range(localSize, nNodes);

        KernelNDRange range(2);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::computeBestSplit(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
    const services::internal::Buffer<algorithmFPType> & response, UniversalBuffer & nodeList, UniversalBuffer & binOffsets, UniversalBuffer & impList,
    UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired, size_t nFeatures, size_t nNodes, size_t minObservationsInLeafNode,
    algorithmFPType impurityThreshold)
{
    services::Status status;

    auto & context = services::internal::getDefaultContext();

    // no overflow check is required because of _nNodesGroups and _nodeGroupProps are small constants
    auto nodesGroups = context.allocate(TypeIds::id<int32_t>(), _nNodesGroups * _nodeGroupProps, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto nodeIndices = context.allocate(TypeIds::id<int32_t>(), nNodes, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(
        _treeLevelBuildHelper.splitNodeListOnGroupsBySize(nodeList, nNodes, nodesGroups, _nNodesGroups, _nodeGroupProps, nodeIndices));

    auto nodesGroupsHost = nodesGroups.template get<int32_t>().toHost(ReadWriteMode::readOnly, status);
    DAAL_CHECK_STATUS_VAR(status);

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
            const size_t nPartialHistograms = maxGroupBlocksNum > _minRowsBlocksForMaxPartHistNum ?
                                                  _maxLocalHistograms / 2 :
                                                  maxGroupBlocksNum < _maxLocalHistograms / 4 ? maxGroupBlocksNum : _maxLocalHistograms / 4;

            // mul overflow for nSelectedFeatures * _nMaxBinsAmongFtrs and for nHistBins * _nHistProps were checked before kernel call in compute
            const size_t nHistBins    = nSelectedFeatures * _nMaxBinsAmongFtrs;
            const size_t partHistSize = nHistBins * _nHistProps;

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nGroupNodes, partHistSize);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nGroupNodes * partHistSize, nPartialHistograms);

            const size_t nPHBlockElems   = nGroupNodes * nPartialHistograms * partHistSize;
            const size_t maxPHBlockElems = _maxPartHistCumulativeSize / sizeof(algorithmFPType);
            const size_t nPHBlocks = nPHBlockElems / maxPHBlockElems ? (nPHBlockElems / maxPHBlockElems + !!(nPHBlockElems % maxPHBlockElems)) : 1;

            size_t nBlockNodes = nGroupNodes / nPHBlocks + !!(nGroupNodes % nPHBlocks);

            for (size_t blockIndicesOffset = groupIndicesOffset; blockIndicesOffset < groupIndicesOffset + nGroupNodes;
                 blockIndicesOffset += nBlockNodes)
            {
                nBlockNodes            = services::internal::min<sse2>(nBlockNodes, groupIndicesOffset + nGroupNodes - blockIndicesOffset);
                auto partialHistograms = context.allocate(TypeIds::id<algorithmFPType>(), nBlockNodes * nPartialHistograms * partHistSize, status);
                DAAL_CHECK_STATUS_VAR(status);
                auto nodesHistograms = context.allocate(TypeIds::id<algorithmFPType>(), nBlockNodes * partHistSize, status);
                DAAL_CHECK_STATUS_VAR(status);

                DAAL_CHECK_STATUS_VAR(computePartialHistograms(data, treeOrder, selectedFeatures, nSelectedFeatures, response, nodeList, nodeIndices,
                                                               blockIndicesOffset, binOffsets, _nMaxBinsAmongFtrs, nFeatures, nBlockNodes,
                                                               partialHistograms, nPartialHistograms));
                DAAL_CHECK_STATUS_VAR(reducePartialHistograms(partialHistograms, nodesHistograms, nPartialHistograms, nBlockNodes, nSelectedFeatures,
                                                              _nMaxBinsAmongFtrs, _reduceLocalSizePartHist));

                DAAL_CHECK_STATUS_VAR(computeBestSplitByHistogram(
                    nodesHistograms, selectedFeatures, nSelectedFeatures, nodeList, nodeIndices, blockIndicesOffset, binOffsets, impList,
                    nodeImpDecreaseList, updateImpDecreaseRequired, nBlockNodes, _nMaxBinsAmongFtrs, minObservationsInLeafNode, impurityThreshold));
            }
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
    const services::internal::Buffer<algorithmFPType> & response, UniversalBuffer & nodeList, UniversalBuffer & nodeIndices, size_t nodeIndicesOffset,
    UniversalBuffer & binOffsets, size_t nMaxBinsAmongFtrs, size_t nFeatures, size_t nNodes, UniversalBuffer & partialHistograms,
    size_t nPartialHistograms)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialHistograms);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelComputePartialHistograms;

    {
        DAAL_ASSERT(nSelectedFeatures <= _int32max);
        DAAL_ASSERT(nodeIndicesOffset <= _int32max);
        DAAL_ASSERT(nMaxBinsAmongFtrs <= _int32max);
        DAAL_ASSERT(nFeatures <= _int32max);
        DAAL_ASSERT(response.size() == _nRows);

        DAAL_ASSERT_UNIVERSAL_BUFFER(data, uint32_t, _nRows * _nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int32_t, _nSelectedRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(selectedFeatures, int32_t, nNodes * nSelectedFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, _nFeatures + 1);
        DAAL_ASSERT_UNIVERSAL_BUFFER(nodeList, int32_t, nNodes * TreeLevelRecord<algorithmFPType>::_nNodeSplitProps);
        DAAL_ASSERT_UNIVERSAL_BUFFER(nodeIndices, int32_t, nNodes);
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialHistograms, algorithmFPType,
                                     nNodes * nPartialHistograms * nSelectedFeatures * _nMaxBinsAmongFtrs * _nHistProps);

        context.fill(partialHistograms, (algorithmFPType)0, status);
        DAAL_CHECK_STATUS_VAR(status);

        KernelArguments args(12, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, nodeList, AccessModeIds::read);
        args.set(3, nodeIndices, AccessModeIds::read);
        args.set(4, static_cast<int32_t>(nodeIndicesOffset));
        args.set(5, selectedFeatures, AccessModeIds::read);
        args.set(6, response, AccessModeIds::read);
        args.set(7, binOffsets, AccessModeIds::read);
        args.set(8, static_cast<int32_t>(nMaxBinsAmongFtrs)); // max num of bins among all ftrs
        args.set(9, static_cast<int32_t>(nFeatures));
        args.set(10, partialHistograms, AccessModeIds::write);
        args.set(11, static_cast<int32_t>(nSelectedFeatures));

        size_t localSize = _preferableGroupSize;

        KernelRange local_range(localSize, 1);
        KernelRange global_range(nPartialHistograms * localSize, nNodes);

        KernelNDRange range(2);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
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

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelReducePartialHistograms;

    {
        DAAL_ASSERT(nPartialHistograms <= _int32max);
        DAAL_ASSERT(nSelectedFeatures <= _int32max);
        DAAL_ASSERT(nMaxBinsAmongFtrs <= _int32max);

        DAAL_ASSERT_UNIVERSAL_BUFFER(partialHistograms, algorithmFPType,
                                     nNodes * nPartialHistograms * nSelectedFeatures * _nMaxBinsAmongFtrs * _nHistProps);
        DAAL_ASSERT_UNIVERSAL_BUFFER(histograms, algorithmFPType, nNodes * nSelectedFeatures * _nMaxBinsAmongFtrs * _nHistProps);

        KernelArguments args(5, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, partialHistograms, AccessModeIds::read);
        args.set(1, histograms, AccessModeIds::write);
        args.set(2, static_cast<int32_t>(nPartialHistograms));
        args.set(3, static_cast<int32_t>(nSelectedFeatures));
        args.set(4, static_cast<int32_t>(nMaxBinsAmongFtrs)); // max num of bins among all ftrs

        KernelRange local_range(1, reduceLocalSize, 1);
        // overflow for nMaxBinsAmongFtrs * nSelectedFeatures should be checked in compute
        KernelRange global_range(nMaxBinsAmongFtrs * nSelectedFeatures, reduceLocalSize, nNodes);

        KernelNDRange range(3);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
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
        const algorithmFPType oobError = computeOOBError(t, x, y, nRows, nFeatures, oobIndices, nOOB, oobBuf, status);
        DAAL_CHECK_STATUS_VAR(status);

        if (mdaRequired)
        {
            DAAL_ASSERT(varImp);
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
                const algorithmFPType permOOBError = computeOOBErrorPerm(t, x, y, nRows, nFeatures, oobIndices, permutation.get(), ftr, nOOB, status);
                DAAL_CHECK_STATUS_VAR(status);

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
                                                                                         size_t n, UniversalBuffer oobBuf, services::Status & status)
{
    typedef DFTreeConverter<algorithmFPType, sse2> DFTreeConverterType;
    typename DFTreeConverterType::TreeHelperType mTreeHelper;

    DAAL_ASSERT(x);
    DAAL_ASSERT(y);
    DAAL_ASSERT_UNIVERSAL_BUFFER(indices, int32_t, n);
    DAAL_ASSERT_UNIVERSAL_BUFFER(oobBuf, algorithmFPType, nRows * _nOOBProps);

    auto rowsIndHost = indices.template get<int32_t>().toHost(ReadWriteMode::readOnly, status);
    auto oobBufHost  = oobBuf.template get<algorithmFPType>().toHost(ReadWriteMode::readWrite, status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, algorithmFPType(0));

    //compute prediction error on each OOB row and get its mean online formulae (Welford)
    //TODO: can be threader_for() block

    algorithmFPType mean = algorithmFPType(0);
    for (size_t i = 0; i < n; i++)
    {
        int rowInd = rowsIndHost.get()[i];
        DAAL_ASSERT(rowInd < nRows);
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
    const UniversalBuffer & indices, const int * indicesPerm, const size_t testFtrInd, size_t n, services::Status & status)
{
    typedef DFTreeConverter<algorithmFPType, sse2> DFTreeConverterType;
    typename DFTreeConverterType::TreeHelperType mTreeHelper;

    DAAL_ASSERT(x);
    DAAL_ASSERT(y);
    DAAL_ASSERT(indicesPerm);
    DAAL_ASSERT(testFtrInd < nFeatures);
    DAAL_ASSERT_UNIVERSAL_BUFFER(indices, int32_t, n);

    auto rowsIndHost = indices.template get<int32_t>().toHost(ReadWriteMode::readOnly, status);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, algorithmFPType(0));

    TArray<algorithmFPType, sse2> buf(nFeatures);
    DAAL_CHECK_COND_ERROR(buf.get(), status, services::ErrorMemoryAllocationFailed);
    DAAL_CHECK_STATUS_RETURN_IF_FAIL(status, algorithmFPType(0));

    algorithmFPType mean = algorithmFPType(0);
    for (size_t i = 0; i < n; i++)
    {
        int rowInd     = rowsIndHost.get()[i];
        int rowIndPerm = indicesPerm[i];
        DAAL_ASSERT(rowInd < nRows);
        DAAL_ASSERT(rowIndPerm < nRows);
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
    services::Status status;

    DAAL_ASSERT(y);
    DAAL_ASSERT_UNIVERSAL_BUFFER(oobBuf, algorithmFPType, nRows * _nOOBProps);

    auto oobBufHost = oobBuf.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly, status);
    DAAL_CHECK_STATUS_VAR(status);

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

    if (res) *res = (0 < nPredicted) ? _res / algorithmFPType(nPredicted) : 0;

    return status;
}

template <typename algorithmFPType>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, hist>::finalizeVarImp(const Parameter & par, algorithmFPType * varImp,
                                                                                         algorithmFPType * varImpVariance, size_t nFeatures)
{
    if (par.varImportance == decision_forest::training::MDA_Scaled)
    {
        if (par.nTrees > 1)
        {
            DAAL_ASSERT(varImpVariance);
            const algorithmFPType div = algorithmFPType(1) / algorithmFPType(par.nTrees);
            for (size_t i = 0; i < nFeatures; i++)
            {
                varImpVariance[i] *= div;
                if (varImpVariance[i] > algorithmFPType(0)) varImp[i] /= daal::internal::Math<algorithmFPType, sse2>::sSqrt(varImpVariance[i] * div);
            }
        }
        else
        {
            DAAL_ASSERT(varImp);
            for (size_t i = 0; i < nFeatures; i++)
            {
                varImp[i] = algorithmFPType(0);
            }
        }
    }
    else if (par.varImportance == decision_forest::training::MDI)
    {
        DAAL_ASSERT(varImp);
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

    _nRows     = x->getNumberOfRows();
    _nFeatures = x->getNumberOfColumns();
    DAAL_CHECK_EX((par.minObservationsInLeafNode <= _int32max), ErrorIncorrectParameter, ParameterName, minObservationsInLeafNodeStr());
    DAAL_CHECK_EX((par.featuresPerNode <= _int32max), ErrorIncorrectParameter, ParameterName, featuresPerNodeStr());
    DAAL_CHECK_EX((par.maxBins <= _int32max), ErrorIncorrectParameter, ParameterName, maxBinsStr());
    DAAL_CHECK_EX((par.minBinSize <= _int32max), ErrorIncorrectParameter, ParameterName, minBinSizeStr());

    if (_nRows > _int32max)
    {
        return services::Status(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
    }
    if (_nFeatures > _int32max)
    {
        return services::Status(services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    }

    const size_t nSelectedFeatures = par.featuresPerNode ? par.featuresPerNode : (_nFeatures > 3 ? _nFeatures / 3 : 1);

    const bool mdaRequired(par.varImportance == decision_forest::training::MDA_Raw || par.varImportance == decision_forest::training::MDA_Scaled);
    const bool oobRequired =
        (par.resultsToCompute & (decision_forest::training::computeOutOfBagError | decision_forest::training::computeOutOfBagErrorPerObservation)
         || mdaRequired);

    decision_forest::regression::internal::ModelImpl & mdImpl =
        *static_cast<daal::algorithms::decision_forest::regression::internal::ModelImpl *>(&m);
    DAAL_CHECK_MALLOC(mdImpl.resize(par.nTrees));

    services::String buildOptions = getBuildOptions();
    DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.init(buildOptions.c_str(), TreeLevel::_nNodeSplitProps));

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory, "part1", df_batch_regression_kernels_part1, buildOptions.c_str()));
    kernelComputeBestSplitSinglePass = kernel_factory.getKernel("computeBestSplitSinglePass", status);

    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory, "part2", df_batch_regression_kernels_part2, buildOptions.c_str()));
    kernelComputeBestSplitByHistogram = kernel_factory.getKernel("computeBestSplitByHistogram", status);
    kernelComputePartialHistograms    = kernel_factory.getKernel("computePartialHistograms", status);
    kernelReducePartialHistograms     = kernel_factory.getKernel("reducePartialHistograms", status);
    DAAL_CHECK_STATUS_VAR(status);

    dtrees::internal::BinParams prm(par.maxBins, par.minBinSize);
    decision_forest::internal::IndexedFeaturesOneAPI<algorithmFPType> indexedFeatures;
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK_MALLOC(featTypes.init(*x));
    DAAL_CHECK_STATUS(status, (indexedFeatures.init(*const_cast<NumericTable *>(x), &featTypes, &prm)));

    _totalBins = indexedFeatures.totalBins();
    /* calculating the maximal number of bins for feature among all features */
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(indexedFeatures.binOffsets(), uint32_t, _nFeatures + 1);
        auto binOffsetsHost = indexedFeatures.binOffsets().template get<uint32_t>().toHost(ReadWriteMode::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);

        _nMaxBinsAmongFtrs = 0;
        for (size_t i = 0; i < _nFeatures; i++)
        {
            auto nFtrBins      = static_cast<size_t>(binOffsetsHost.get()[i + 1] - binOffsetsHost.get()[i]);
            _nMaxBinsAmongFtrs = (_nMaxBinsAmongFtrs < nFtrBins) ? nFtrBins : _nMaxBinsAmongFtrs;
        }
    }
    // no need to check for _nMaxBinsAmongFtrs < INT32_MAX because it will not be bigger than _nRows and _nRows was already checked
    // check mul overflow for _nMaxBinsAmongFtrs * nSelectedFeatures
    // and _nMaxBinsAmongFtrs * nSelectedFeatures * _nHistProps because they are used further in kernels
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nMaxBinsAmongFtrs, nSelectedFeatures);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nMaxBinsAmongFtrs * nSelectedFeatures, _nHistProps);

    _nSelectedRows = par.observationsPerTreeFraction * _nRows;
    DAAL_CHECK_EX((_nSelectedRows > 0), ErrorIncorrectParameter, ParameterName, observationsPerTreeFractionStr());

    daal::services::internal::TArray<int, sse2> selectedRowsHost(_nSelectedRows);
    DAAL_CHECK_MALLOC(selectedRowsHost.get());

    auto treeOrderLev = context.allocate(TypeIds::id<int32_t>(), _nSelectedRows, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto treeOrderLevBuf = context.allocate(TypeIds::id<int32_t>(), _nSelectedRows, status);
    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> dataBlock;
    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->getBlockOfRows(0, _nRows, readOnly, dataBlock));

    /* blocks for varImp MDI calculation */
    bool mdiRequired         = (par.varImportance == decision_forest::training::MDI);
    auto nodeImpDecreaseList = context.allocate(TypeIds::id<algorithmFPType>(), 1, status); // holder will be reallocated in loop
    DAAL_CHECK_STATUS_VAR(status);
    BlockDescriptor<algorithmFPType> varImpBlock;
    NumericTablePtr varImpResPtr = res.get(variableImportance);

    if (mdiRequired || mdaRequired)
    {
        DAAL_CHECK_STATUS_VAR(varImpResPtr->getBlockOfRows(0, 1, writeOnly, varImpBlock));
        context.fill(varImpBlock.getBuffer(), (algorithmFPType)0, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    /* blocks for OutOfBag error calculation */
    UniversalBuffer oobBufferPerObs;
    if (oobRequired)
    {
        // oobBufferPerObs contains pair <cumulative value, count> for all out of bag observations for all trees
        oobBufferPerObs = context.allocate(TypeIds::id<algorithmFPType>(), _nRows * _nOOBProps, status);
        DAAL_CHECK_STATUS_VAR(status);
        context.fill(oobBufferPerObs, algorithmFPType(0), status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    /* blocks for MDA scaled error calculation */
    bool mdaScaledRequired = (par.varImportance == decision_forest::training::MDA_Scaled);
    daal::services::internal::TArrayCalloc<algorithmFPType, sse2> varImpVariance; // for now it is calculated on host
    if (mdaScaledRequired)
    {
        varImpVariance.reset(_nFeatures);
    }

    /*init engines*/
    engines::internal::ParallelizationTechnique technique = engines::internal::family;
    selectParallelizationTechnique<sse2>(par, technique);
    engines::internal::Params<sse2> params(par.nTrees);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par.nTrees - 1, par.nTrees);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (par.nTrees - 1) * par.nTrees, _nRows);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (par.nTrees - 1) * par.nTrees * _nRows, (par.featuresPerNode + 1));
    for (size_t i = 0; i < par.nTrees; i++)
    {
        params.nSkip[i] = i * par.nTrees * _nRows * (par.featuresPerNode + 1);
    }
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par.nTrees, sizeof(engines::EnginePtr));
    daal::services::internal::TArray<engines::EnginePtr, sse2> engines(par.nTrees);
    engines::internal::EnginesCollection<sse2> enginesCollection(par.engine, technique, params, engines, &status);
    DAAL_CHECK_STATUS_VAR(status);

    for (size_t iter = 0; (iter < par.nTrees) && !algorithms::internal::isCancelled(status, pHostApp); ++iter)
    {
        BlockDescriptor<algorithmFPType> responseBlock;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(y)->getBlockOfRows(0, _nRows, readOnly, responseBlock));

        size_t nNodes   = 1; // num of potential nodes to split on current tree level
        size_t nOOBRows = 0;

        Collection<TreeLevel> DFTreeRecords;
        Collection<UniversalBuffer> levelNodeLists;    // lists of nodes int props(rowsOffset, rows, ftrId, ftrVal ... )
        Collection<UniversalBuffer> levelNodeImpLists; // list of nodes fptype props (impurity, mean)
        UniversalBuffer oobRows;

        // no check for overflow required because nNodes = 1, splitProps and impProps are small constants
        levelNodeLists.push_back(context.allocate(TypeIds::id<int32_t>(), nNodes * TreeLevel::_nNodeSplitProps, status));
        DAAL_CHECK_STATUS_VAR(status);
        levelNodeImpLists.push_back(context.allocate(TypeIds::id<algorithmFPType>(), nNodes * TreeLevel::_nNodeImpProps, status));
        DAAL_CHECK_STATUS_VAR(status);

        {
            auto rootNode = levelNodeLists[0].template get<int32_t>().toHost(ReadWriteMode::writeOnly, status);
            DAAL_CHECK_STATUS_VAR(status);
            rootNode.get()[0] = 0;              // rows offset
            rootNode.get()[1] = _nSelectedRows; // num of rows
        }

        auto engineImpl = dynamic_cast<engines::internal::BatchBaseImpl *>(engines[iter].get());
        if (!engineImpl) return Status(ErrorEngineNotSupported);

        if (par.bootstrap)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.RNG);
            daal::internal::RNGs<int, sse2> rng;
            rng.uniform(_nSelectedRows, selectedRowsHost.get(), engineImpl->getState(), 0, _nRows);

            context.copy(treeOrderLev, 0, (void *)selectedRowsHost.get(), _nSelectedRows, 0, _nSelectedRows, status);
            DAAL_CHECK_STATUS_VAR(status);
        }
        else
        {
            DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.initializeTreeOrder(_nSelectedRows, treeOrderLev));
        }

        if (oobRequired)
        {
            _treeLevelBuildHelper.getOOBRows(treeOrderLev, _nSelectedRows, nOOBRows, oobRows); // nOOBRows and oobRows are the output
        }

        for (size_t level = 0; nNodes > 0; level++)
        {
            auto nodeList = levelNodeLists[level];
            auto impList  = levelNodeImpLists[level];

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (nNodes + 1), nSelectedFeatures);
            daal::services::internal::TArray<int, sse2> selectedFeaturesHost(
                (nNodes + 1) * nSelectedFeatures); // first part is used features indices, +1 - part for generator
            DAAL_CHECK_MALLOC(selectedFeaturesHost.get());

            auto selectedFeaturesCom = context.allocate(TypeIds::id<int32_t>(), nNodes * nSelectedFeatures, status);
            DAAL_CHECK_STATUS_VAR(status);

            if (nSelectedFeatures != _nFeatures)
            {
                daal::internal::RNGs<int, sse2> rng;
                for (size_t node = 0; node < nNodes; node++)
                {
                    rng.uniformWithoutReplacement(nSelectedFeatures, selectedFeaturesHost.get() + node * nSelectedFeatures,
                                                  selectedFeaturesHost.get() + (node + 1) * nSelectedFeatures, engineImpl->getState(), 0, _nFeatures);
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

            context.copy(selectedFeaturesCom, 0, (void *)selectedFeaturesHost.get(), nSelectedFeatures * nNodes, 0, nSelectedFeatures * nNodes,
                         status);
            DAAL_CHECK_STATUS_VAR(status);

            if (mdiRequired)
            {
                nodeImpDecreaseList = context.allocate(TypeIds::id<algorithmFPType>(), nNodes, status);
                DAAL_CHECK_STATUS_VAR(status);
            }

            DAAL_CHECK_STATUS_VAR(computeBestSplit(indexedFeatures.getFullData(), treeOrderLev, selectedFeaturesCom, nSelectedFeatures,
                                                   responseBlock.getBuffer(), nodeList, indexedFeatures.binOffsets(), impList, nodeImpDecreaseList,
                                                   mdiRequired, _nFeatures, nNodes, par.minObservationsInLeafNode, par.impurityThreshold));

            if (par.maxTreeDepth > 0 && par.maxTreeDepth == level)
            {
                DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.convertSplitToLeaf(nodeList, nNodes));
                TreeLevel levelRecord;
                DAAL_CHECK_STATUS_VAR(levelRecord.init(nodeList, impList, nNodes));
                DFTreeRecords.push_back(levelRecord);
                break;
            }

            TreeLevel levelRecord;
            DAAL_CHECK_STATUS_VAR(levelRecord.init(nodeList, impList, nNodes));
            DFTreeRecords.push_back(levelRecord);

            if (mdiRequired)
            {
                /*mdi is calculated only on split nodes and not calculated on last level*/
                auto varImpBuffer = varImpBlock.getBuffer();
                DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.updateMDIVarImportance(nodeList, nodeImpDecreaseList, nNodes, varImpBuffer, _nFeatures));
            }

            size_t nNodesNewLevel;
            DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.getNumOfSplitNodes(nodeList, nNodes, nNodesNewLevel));

            if (nNodesNewLevel)
            {
                /*there are split nodes -> next level is required*/
                nNodesNewLevel *= 2;

                DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nNodesNewLevel, TreeLevel::_nNodeSplitProps);
                DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nNodesNewLevel, (TreeLevel::_nNodeImpProps));
                auto nodeListNewLevel = context.allocate(TypeIds::id<int32_t>(), nNodesNewLevel * TreeLevel::_nNodeSplitProps, status);
                DAAL_CHECK_STATUS_VAR(status);
                auto impListNewLevel = context.allocate(TypeIds::id<algorithmFPType>(), nNodesNewLevel * TreeLevel::_nNodeImpProps, status);
                DAAL_CHECK_STATUS_VAR(status);

                DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.doNodesSplit(nodeList, nNodes, nodeListNewLevel, nNodesNewLevel));

                levelNodeLists.push_back(nodeListNewLevel);
                levelNodeImpLists.push_back(impListNewLevel);

                DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.doLevelPartition(indexedFeatures.getFullData(), nodeList, nNodes, treeOrderLev,
                                                                             treeOrderLevBuf, _nSelectedRows, _nFeatures));
            }

            nNodes = nNodesNewLevel;
        } // for level

        services::Collection<SharedPtr<algorithmFPType> > binValuesHost(_nFeatures);
        DAAL_CHECK_MALLOC(binValuesHost.data());
        services::Collection<algorithmFPType *> binValues(_nFeatures);
        DAAL_CHECK_MALLOC(binValues.data());

        for (size_t i = 0; i < _nFeatures; i++)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER(indexedFeatures.binBorders(i), algorithmFPType, indexedFeatures.numIndices(i));
            binValuesHost[i] = indexedFeatures.binBorders(i).template get<algorithmFPType>().toHost(ReadWriteMode::readOnly, status);
            DAAL_CHECK_STATUS_VAR(status);
            binValues[i] = binValuesHost[i].get();
        }

        typename DFTreeConverterType::TreeHelperType mTreeHelper;

        DFTreeConverterType converter;
        DAAL_CHECK_STATUS_VAR(converter.convertToDFDecisionTree(DFTreeRecords, binValues.data(), mTreeHelper));

        mdImpl.add(mTreeHelper._tree, 0 /*nClasses*/, iter);

        DAAL_CHECK_STATUS_VAR(computeResults(mTreeHelper._tree, dataBlock.getBlockPtr(), responseBlock.getBlockPtr(), _nSelectedRows, _nFeatures,
                                             oobRows, nOOBRows, oobBufferPerObs, varImpBlock.getBlockPtr(), varImpVariance.get(), iter + 1,
                                             engines[iter], par));

        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(y)->releaseBlockOfRows(responseBlock));
    }

    /* Finalize results */
    if (par.resultsToCompute & (decision_forest::training::computeOutOfBagError | decision_forest::training::computeOutOfBagErrorPerObservation))
    {
        BlockDescriptor<algorithmFPType> responseBlock;
        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(y)->getBlockOfRows(0, _nRows, readOnly, responseBlock));

        NumericTablePtr oobErrPtr = res.get(outOfBagError);
        BlockDescriptor<algorithmFPType> oobErrBlock;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagError)
            DAAL_CHECK_STATUS_VAR(oobErrPtr->getBlockOfRows(0, 1, writeOnly, oobErrBlock));

        NumericTablePtr oobErrPerObsPtr = res.get(outOfBagErrorPerObservation);
        BlockDescriptor<algorithmFPType> oobErrPerObsBlock;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
            DAAL_CHECK_STATUS_VAR(oobErrPerObsPtr->getBlockOfRows(0, _nRows, writeOnly, oobErrPerObsBlock));

        DAAL_CHECK_STATUS_VAR(
            finalizeOOBError(responseBlock.getBlockPtr(), oobBufferPerObs, _nRows, oobErrBlock.getBlockPtr(), oobErrPerObsBlock.getBlockPtr()));

        if (oobErrPtr) DAAL_CHECK_STATUS_VAR(oobErrPtr->releaseBlockOfRows(oobErrBlock));

        if (oobErrPerObsPtr) DAAL_CHECK_STATUS_VAR(oobErrPerObsPtr->releaseBlockOfRows(oobErrPerObsBlock));

        DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(y)->releaseBlockOfRows(responseBlock));
    }

    if (par.varImportance != decision_forest::training::none && par.varImportance != decision_forest::training::MDA_Raw)
    {
        DAAL_CHECK_STATUS_VAR(finalizeVarImp(par, varImpBlock.getBlockPtr(), varImpVariance.get(), _nFeatures));
    }

    if (mdiRequired || mdaRequired) DAAL_CHECK_STATUS_VAR(varImpResPtr->releaseBlockOfRows(varImpBlock));

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
