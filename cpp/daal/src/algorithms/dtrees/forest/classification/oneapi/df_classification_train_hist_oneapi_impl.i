/* file: df_classification_train_hist_oneapi_impl.i */
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
//  Implementation of auxiliary functions for decision forest classification
//  hist method.
//--
*/

#ifndef __DF_CLASSIFICATION_TRAIN_HIST_ONEAPI_IMPL_I__
#define __DF_CLASSIFICATION_TRAIN_HIST_ONEAPI_IMPL_I__

#include "src/algorithms/dtrees/forest/classification/oneapi/df_classification_train_hist_kernel_oneapi.h"
#include "src/algorithms/dtrees/forest/classification/oneapi/cl_kernels/df_batch_classification_kernels.cl"

#include "src/algorithms/dtrees/forest/oneapi/df_feature_type_helper_oneapi.i"
#include "src/algorithms/dtrees/forest/oneapi/df_tree_level_build_helper_oneapi.i"
#include "src/algorithms/dtrees/forest/classification/df_classification_model_impl.h"
#include "src/algorithms/dtrees/forest/classification/oneapi/df_classification_tree_helper_impl.i"

#include "src/externals/service_ittnotify.h"
#include "src/externals/service_rng.h"
#include "src/externals/service_math.h" //will remove after migrating finalize MDA to GPU
#include "services/buffer.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "services/error_indexes.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_algo_utils.h"
#include "src/services/service_arrays.h"
#include "src/services/service_utils.h"
#include "src/services/daal_strings.h"
#include "src/algorithms/engines/engine_types_internal.h"
#include "sycl/internal/types.h"

using namespace daal::algorithms::decision_forest::internal;
using namespace daal::algorithms::decision_forest::classification::internal;

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

static services::String getBuildOptions(size_t nClasses)
{
    DAAL_ASSERT(nClasses <= _int32max);
    char buffer[DAAL_MAX_STRING_SIZE] = { 0 };
    const auto written                = daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, static_cast<int32_t>(nClasses));
    services::String nClassesStr(buffer, written);

    services::String buildOptions = " -D NODE_PROPS=6 -D IMPURITY_PROPS=1 -D HIST_PROPS=";
    buildOptions.add(nClassesStr);
    buildOptions.add(" -D NUM_OF_CLASSES=");
    buildOptions.add(nClassesStr);

    return buildOptions;
}

template <typename algorithmFPType>
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::buildProgram(ClKernelFactoryIface & factory, const char * programName,
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

        services::String cachekey("__daal_algorithms_df_batch_classification_");
        cachekey.add(build_options);
        cachekey.add(programName);

        factory.build(ExecutionTargetIds::device, cachekey.c_str(), programSrc, build_options.c_str());
    }

    return status;
}

template <typename algorithmFPType>
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::computeBestSplitByHistogram(
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
        DAAL_ASSERT(nSelectedFeatures <= _int32max);
        DAAL_ASSERT(nodeIndicesOffset <= _int32max);
        DAAL_ASSERT(updateImpDecreaseRequired <= _int32max);
        DAAL_ASSERT(nMaxBinsAmongFtrs <= _int32max);
        DAAL_ASSERT(minObservationsInLeafNode <= _int32max);

        KernelArguments args(13);
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
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::computeBestSplitSinglePass(
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
        DAAL_ASSERT(nSelectedFeatures <= _int32max);
        DAAL_ASSERT(nodeIndicesOffset <= _int32max);
        DAAL_ASSERT(updateImpDecreaseRequired <= _int32max);
        DAAL_ASSERT(nFeatures <= _int32max);
        DAAL_ASSERT(minObservationsInLeafNode <= _int32max);

        KernelArguments args(15);
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
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::computeBestSplit(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & selectedFeatures, size_t nSelectedFeatures,
    const services::Buffer<algorithmFPType> & response, UniversalBuffer & nodeList, UniversalBuffer & binOffsets, UniversalBuffer & impList,
    UniversalBuffer & nodeImpDecreaseList, bool updateImpDecreaseRequired, size_t nFeatures, size_t nNodes, size_t minObservationsInLeafNode,
    algorithmFPType impurityThreshold)
{
    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    // no overflow check is required because of _nNodesGroups and _nodeGroupProps are small constants
    auto nodesGroups = context.allocate(TypeIds::id<int32_t>(), _nNodesGroups * _nodeGroupProps, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto nodeIndices = context.allocate(TypeIds::id<int32_t>(), nNodes, &status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.splitNodeListOnGroupsBySize(nodeList, nNodes, nodesGroups, nodeIndices));

    auto nodesGroupsHost = nodesGroups.template get<int32_t>().toHost(ReadWriteMode::readOnly);
    DAAL_CHECK_MALLOC(nodesGroupsHost.get());

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
            int reduceLocalSize = 16; // add logic for its adjustment

            // mul overflow for nSelectedFeatures * _nMaxBinsAmongFtrs and for nHistBins * _nClasses were checked before kernel call in compute
            size_t nHistBins    = nSelectedFeatures * _nMaxBinsAmongFtrs;
            size_t partHistSize = nHistBins * _nClasses;

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nGroupNodes, partHistSize);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nGroupNodes * partHistSize, nPartialHistograms);

            auto partialHistograms = context.allocate(TypeIds::id<algorithmFPType>(), nGroupNodes * nPartialHistograms * partHistSize, &status);
            DAAL_CHECK_STATUS_VAR(status);
            auto nodesHistograms = context.allocate(TypeIds::id<algorithmFPType>(), nGroupNodes * partHistSize, &status);
            DAAL_CHECK_STATUS_VAR(status);

            DAAL_CHECK_STATUS_VAR(computePartialHistograms(data, treeOrder, selectedFeatures, nSelectedFeatures, response, nodeList, nodeIndices,
                                                           groupIndicesOffset, binOffsets, _nMaxBinsAmongFtrs, nFeatures, nGroupNodes,
                                                           partialHistograms, nPartialHistograms));

            DAAL_CHECK_STATUS_VAR(reducePartialHistograms(partialHistograms, nodesHistograms, nPartialHistograms, nGroupNodes, nSelectedFeatures,
                                                          _nMaxBinsAmongFtrs, reduceLocalSize));

            DAAL_CHECK_STATUS_VAR(computeBestSplitByHistogram(nodesHistograms, selectedFeatures, nSelectedFeatures, nodeList, nodeIndices,
                                                              groupIndicesOffset, binOffsets, impList, nodeImpDecreaseList, updateImpDecreaseRequired,
                                                              nGroupNodes, _nMaxBinsAmongFtrs, minObservationsInLeafNode, impurityThreshold));
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
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::computePartialHistograms(
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
        DAAL_ASSERT(nSelectedFeatures <= _int32max);
        DAAL_ASSERT(nodeIndicesOffset <= _int32max);
        DAAL_ASSERT(nMaxBinsAmongFtrs <= _int32max);
        DAAL_ASSERT(nFeatures <= _int32max);

        KernelArguments args(11);
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
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::reducePartialHistograms(
    UniversalBuffer & partialHistograms, UniversalBuffer & histograms, size_t nPartialHistograms, size_t nNodes, size_t nSelectedFeatures,
    size_t nMaxBinsAmongFtrs, size_t reduceLocalSize)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.reducePartialHistograms);

    services::Status status;

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    auto & kernel = kernelReducePartialHistograms;

    {
        DAAL_ASSERT(nPartialHistograms <= _int32max);
        DAAL_ASSERT(nSelectedFeatures <= _int32max);
        DAAL_ASSERT(nMaxBinsAmongFtrs <= _int32max);

        KernelArguments args(5);
        args.set(0, partialHistograms, AccessModeIds::read);
        args.set(1, histograms, AccessModeIds::write);
        args.set(2, static_cast<int32_t>(nPartialHistograms));
        args.set(3, static_cast<int32_t>(nSelectedFeatures));
        args.set(4, static_cast<int32_t>(nMaxBinsAmongFtrs)); // max num of bins among all ftrs

        KernelRange local_range(1, reduceLocalSize, 1);
        // overflow for nMaxBinsAmongFtrs * nSelectedFeatures should be checked in compute
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
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::computeResults(
    const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, size_t nRows, size_t nFeatures,
    const UniversalBuffer & oobIndices, size_t nOOB, UniversalBuffer & oobBuf, algorithmFPType * varImp, algorithmFPType * varImpVariance,
    size_t nBuiltTrees, const engines::EnginePtr & engine, const Parameter & par)
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
algorithmFPType ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::computeOOBError(const dtrees::internal::Tree & t,
                                                                                             const algorithmFPType * x, const algorithmFPType * y,
                                                                                             const size_t nRows, const size_t nFeatures,
                                                                                             const UniversalBuffer & indices, size_t n,
                                                                                             UniversalBuffer oobBuf, services::Status * status)
{
    typedef DFTreeConverter<algorithmFPType, sse2> DFTreeConverterType;
    typename DFTreeConverterType::TreeHelperType mTreeHelper;

    auto rowsIndHost = indices.template get<int32_t>().toHost(ReadWriteMode::readOnly);
    DAAL_CHECK_MALLOC(rowsIndHost.get());
    auto oobBufHost = oobBuf.template get<uint32_t>().toHost(ReadWriteMode::readWrite);
    DAAL_CHECK_MALLOC(oobBufHost.get());

    //compute prediction error on each OOB row and get its mean online formulae (Welford)
    //TODO: can be threader_for() block

    algorithmFPType mean = algorithmFPType(0);
    for (size_t i = 0; i < n; i++)
    {
        int rowInd        = rowsIndHost.get()[i];
        size_t prediction = mTreeHelper.predict(t, &x[rowInd * nFeatures]);
        oobBufHost.get()[rowInd * _nClasses + prediction]++;
        algorithmFPType val = algorithmFPType(prediction != size_t(y[rowInd]));
        mean += (val - mean) / algorithmFPType(i + 1);
    }

    return mean;
}

template <typename algorithmFPType>
algorithmFPType ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::computeOOBErrorPerm(
    const dtrees::internal::Tree & t, const algorithmFPType * x, const algorithmFPType * y, const size_t nRows, const size_t nFeatures,
    const UniversalBuffer & indices, const int * indicesPerm, const size_t testFtrInd, size_t n, services::Status * status)
{
    typedef DFTreeConverter<algorithmFPType, sse2> DFTreeConverterType;
    typename DFTreeConverterType::TreeHelperType mTreeHelper;

    auto rowsIndHost = indices.template get<int32_t>().toHost(ReadWriteMode::readOnly);
    DAAL_CHECK_MALLOC(rowsIndHost.get());
    TArray<algorithmFPType, sse2> buf(nFeatures);
    DAAL_CHECK_MALLOC(buf.get());

    algorithmFPType mean = algorithmFPType(0);
    for (size_t i = 0; i < n; i++)
    {
        int rowInd     = rowsIndHost.get()[i];
        int rowIndPerm = indicesPerm[i];
        services::internal::tmemcpy<algorithmFPType, sse2>(buf.get(), &x[rowInd * nFeatures], nFeatures);
        buf[testFtrInd]     = x[rowIndPerm * nFeatures + testFtrInd];
        size_t prediction   = mTreeHelper.predict(t, buf.get());
        algorithmFPType val = algorithmFPType(prediction != size_t(y[rowInd]));
        mean += (val - mean) / algorithmFPType(i + 1);
    }

    return mean;
}

template <typename algorithmFPType>
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::finalizeOOBError(const algorithmFPType * y,
                                                                                               const UniversalBuffer & oobBuf, const size_t nRows,
                                                                                               algorithmFPType * res, algorithmFPType * resPerObs)
{
    auto oobBufHost = oobBuf.template get<uint32_t>().toHost(ReadWriteMode::readOnly);
    DAAL_CHECK_MALLOC(oobBufHost.get());

    size_t nPredicted    = 0;
    algorithmFPType _res = 0;

    for (size_t i = 0; i < nRows; i++)
    {
        size_t prediction = 0;
        size_t expectation(y[i]);
        size_t maxVal = 0;
        for (size_t clsIdx = 0; clsIdx < _nClasses; clsIdx++)
        {
            size_t val = oobBufHost.get()[i * _nClasses + clsIdx];
            if (val > maxVal)
            {
                maxVal     = val;
                prediction = clsIdx;
            }
        }

        if (0 < maxVal)
        {
            algorithmFPType predictionRes = algorithmFPType(prediction != expectation);
            if (resPerObs) resPerObs[i] = predictionRes;
            _res += predictionRes;
            nPredicted++;
        }
        else if (resPerObs)
            resPerObs[i] = algorithmFPType(-1); //was not in OOB set of any tree and hence not predicted
    }

    if (res) *res = (0 < nPredicted) ? _res / algorithmFPType(nPredicted) : 0;

    return services::Status();
}

template <typename algorithmFPType>
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::finalizeVarImp(const Parameter & par, algorithmFPType * varImp,
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
/* compute method for ClassificationTrainBatchKernelOneAPI */
///////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType>
services::Status ClassificationTrainBatchKernelOneAPI<algorithmFPType, hist>::compute(HostAppIface * pHostApp, const NumericTable * x,
                                                                                      const NumericTable * y,
                                                                                      decision_forest::classification::Model & m, Result & res,
                                                                                      const Parameter & par)
{
    typedef DFTreeConverter<algorithmFPType, sse2> DFTreeConverterType;
    typedef TreeLevelRecord<algorithmFPType> TreeLevel;

    services::Status status;

    _nClasses = par.nClasses;

    const size_t nRows     = x->getNumberOfRows();
    const size_t nFeatures = x->getNumberOfColumns();

    DAAL_CHECK_EX((par.nClasses <= _int32max), ErrorIncorrectParameter, ParameterName, nClassesStr());
    DAAL_CHECK_EX((par.minObservationsInLeafNode <= _int32max), ErrorIncorrectParameter, ParameterName, minObservationsInLeafNodeStr());
    DAAL_CHECK_EX((par.featuresPerNode <= _int32max), ErrorIncorrectParameter, ParameterName, featuresPerNodeStr());

    if (nRows > _int32max)
    {
        return services::Status(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
    }
    if (nFeatures > _int32max)
    {
        return services::Status(services::ErrorIncorrectNumberOfColumnsInInputNumericTable);
    }

    const size_t nSelectedFeatures = par.featuresPerNode ? par.featuresPerNode : daal::internal::Math<algorithmFPType, sse2>::sSqrt(nFeatures);

    const bool mdaRequired(par.varImportance == decision_forest::training::MDA_Raw || par.varImportance == decision_forest::training::MDA_Scaled);
    const bool oobRequired =
        (par.resultsToCompute & (decision_forest::training::computeOutOfBagError | decision_forest::training::computeOutOfBagErrorPerObservation)
         || mdaRequired);

    decision_forest::classification::internal::ModelImpl & mdImpl =
        *static_cast<daal::algorithms::decision_forest::classification::internal::ModelImpl *>(&m);
    DAAL_CHECK_MALLOC(mdImpl.resize(par.nTrees));

    services::String buildOptions = getBuildOptions(_nClasses);
    DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.init(buildOptions.c_str()));

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory, "part1", df_batch_classification_kernels_part1, buildOptions.c_str()));
    kernelComputeBestSplitSinglePass = kernel_factory.getKernel("computeBestSplitSinglePass", &status);

    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory, "part2", df_batch_classification_kernels_part2, buildOptions.c_str()));
    kernelComputeBestSplitByHistogram = kernel_factory.getKernel("computeBestSplitByHistogram", &status);
    kernelComputePartialHistograms    = kernel_factory.getKernel("computePartialHistograms", &status);
    kernelReducePartialHistograms     = kernel_factory.getKernel("reducePartialHistograms", &status);
    DAAL_CHECK_STATUS_VAR(status);

    dtrees::internal::BinParams prm(_maxBins, par.minObservationsInLeafNode);
    decision_forest::internal::IndexedFeaturesOneAPI<algorithmFPType> indexedFeatures;
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK_MALLOC(featTypes.init(*x));
    DAAL_CHECK_STATUS(status, (indexedFeatures.init(*const_cast<NumericTable *>(x), &featTypes, &prm)));

    _totalBins = indexedFeatures.totalBins();
    /* calculating the maximal number of bins for feature among all features */
    {
        auto binOffsetsHost = indexedFeatures.binOffsets().template get<int32_t>().toHost(ReadWriteMode::readOnly);
        DAAL_CHECK_MALLOC(binOffsetsHost.get());
        _nMaxBinsAmongFtrs = 0;
        for (size_t i = 0; i < nFeatures; i++)
        {
            auto nFtrBins      = static_cast<size_t>(binOffsetsHost.get()[i + 1] - binOffsetsHost.get()[i]);
            _nMaxBinsAmongFtrs = (_nMaxBinsAmongFtrs < nFtrBins) ? nFtrBins : _nMaxBinsAmongFtrs;
        }
    }
    // no need to check for _nMaxBinsAmongFtrs < INT32_MAX because it will not be bigger than nRows and nRows was already checked
    // check mul overflow for _nMaxBinsAmongFtrs * nSelectedFeatures
    // and _nMaxBinsAmongFtrs * nSelectedFeatures * _nClasses because they are used further in kernels
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nMaxBinsAmongFtrs, nSelectedFeatures);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _nMaxBinsAmongFtrs * nSelectedFeatures, _nClasses);

    const size_t nSelectedRows = par.observationsPerTreeFraction * nRows;
    DAAL_CHECK_EX((nSelectedRows > 0), ErrorIncorrectParameter, ParameterName, observationsPerTreeFractionStr());

    daal::services::internal::TArray<int, sse2> selectedRowsHost(nSelectedRows);
    DAAL_CHECK_MALLOC(selectedRowsHost.get());

    auto treeOrderLev = context.allocate(TypeIds::id<int32_t>(), nSelectedRows, &status);
    DAAL_CHECK_STATUS_VAR(status);
    auto treeOrderLevBuf = context.allocate(TypeIds::id<int32_t>(), nSelectedRows, &status);
    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> dataBlock;
    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->getBlockOfRows(0, nRows, readOnly, dataBlock));

    /* blocks for varImp MDI calculation */
    bool mdiRequired         = (par.varImportance == decision_forest::training::MDI);
    auto nodeImpDecreaseList = context.allocate(TypeIds::id<algorithmFPType>(), 1, &status); // holder will be reallocated in loop
    DAAL_CHECK_STATUS_VAR(status);
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
        // oobBufferPerObs contains nClassed counters for all out of bag observations for all trees
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, _nClasses);
        oobBufferPerObs = context.allocate(TypeIds::id<uint32_t>(), nRows * _nClasses, &status);
        DAAL_CHECK_STATUS_VAR(status);
        context.fill(oobBufferPerObs, 0, &status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    /* blocks for MDA scaled error calculation */
    bool mdaScaledRequired = (par.varImportance == decision_forest::training::MDA_Scaled);
    daal::services::internal::TArrayCalloc<algorithmFPType, sse2> varImpVariance; // for now it is calculated on host
    if (mdaScaledRequired)
    {
        varImpVariance.reset(nFeatures);
    }

    /*init engines*/
    engines::internal::ParallelizationTechnique technique = engines::internal::family;
    selectParallelizationTechnique<sse2>(par, technique);
    engines::internal::Params<sse2> params(par.nTrees);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par.nTrees - 1, par.nTrees);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (par.nTrees - 1) * par.nTrees, nRows);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (par.nTrees - 1) * par.nTrees * nRows, (par.featuresPerNode + 1));
    for (size_t i = 0; i < par.nTrees; i++)
    {
        params.nSkip[i] = i * par.nTrees * nRows * (par.featuresPerNode + 1);
    }
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par.nTrees, sizeof(engines::EnginePtr));
    daal::services::internal::TArray<engines::EnginePtr, sse2> engines(par.nTrees);
    engines::internal::EnginesCollection<sse2> enginesCollection(par.engine, technique, params, engines, &status);
    DAAL_CHECK_STATUS_VAR(status);

    if (!par.bootstrap)
    {
        DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.initializeTreeOrder(nSelectedRows, treeOrderLev));
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

        // no check for overflow required because nNodes = 1, splitProps and impProps are small constants
        levelNodeLists.push_back(context.allocate(TypeIds::id<int32_t>(), nNodes * TreeLevel::_nNodeSplitProps, &status));
        DAAL_CHECK_STATUS_VAR(status);
        levelNodeImpLists.push_back(context.allocate(TypeIds::id<algorithmFPType>(), nNodes * (TreeLevel::_nNodeImpProps + _nClasses), &status));
        DAAL_CHECK_STATUS_VAR(status);

        {
            auto rootNode = levelNodeLists[0].template get<int32_t>().toHost(ReadWriteMode::writeOnly);
            DAAL_CHECK_MALLOC(rootNode.get());
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
            _treeLevelBuildHelper.getOOBRows(treeOrderLev, nSelectedRows, nOOBRows, oobRows); // nOOBRows and oobRows are the output
        }

        for (size_t level = 0; nNodes > 0; level++)
        {
            auto nodeList = levelNodeLists[level];
            auto impList  = levelNodeImpLists[level];

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, (nNodes + 1), nSelectedFeatures);
            daal::services::internal::TArray<int, sse2> selectedFeaturesHost(
                (nNodes + 1) * nSelectedFeatures); // first part is used features indices, +1 - part for generator
            DAAL_CHECK_MALLOC(selectedFeaturesHost.get());

            auto selectedFeaturesCom = context.allocate(TypeIds::id<int32_t>(), nNodes * nSelectedFeatures, &status);
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
                DAAL_CHECK_STATUS_VAR(status);
            }

            DAAL_CHECK_STATUS_VAR(computeBestSplit(indexedFeatures.getFullData(), treeOrderLev, selectedFeaturesCom, nSelectedFeatures,
                                                   responseBlock.getBuffer(), nodeList, indexedFeatures.binOffsets(), impList, nodeImpDecreaseList,
                                                   mdiRequired, nFeatures, nNodes, par.minObservationsInLeafNode, par.impurityThreshold));

            if (par.maxTreeDepth > 0 && par.maxTreeDepth == level)
            {
                DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.convertSplitToLeaf(nodeList, nNodes));
                TreeLevel levelRecord;
                DAAL_CHECK_STATUS_VAR(levelRecord.init(nodeList, impList, nNodes, _nClasses));
                DFTreeRecords.push_back(levelRecord);
                break;
            }

            TreeLevel levelRecord;
            DAAL_CHECK_STATUS_VAR(levelRecord.init(nodeList, impList, nNodes, _nClasses));
            DFTreeRecords.push_back(levelRecord);

            if (mdiRequired)
            {
                /*mdi is calculated only on split nodes and not calculated on last level*/
                auto varImpBuffer = varImpBlock.getBuffer();
                DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.updateMDIVarImportance(nodeList, nodeImpDecreaseList, nNodes, varImpBuffer, nFeatures));
            }

            size_t nNodesNewLevel;
            DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.getNumOfSplitNodes(nodeList, nNodes, nNodesNewLevel));

            if (nNodesNewLevel)
            {
                /*there are split nodes -> next level is required*/
                nNodesNewLevel *= 2;

                DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nNodesNewLevel, TreeLevel::_nNodeSplitProps);
                DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nNodesNewLevel, (TreeLevel::_nNodeImpProps + _nClasses));
                auto nodeListNewLevel = context.allocate(TypeIds::id<int32_t>(), nNodesNewLevel * TreeLevel::_nNodeSplitProps, &status);
                DAAL_CHECK_STATUS_VAR(status);
                auto impListNewLevel =
                    context.allocate(TypeIds::id<algorithmFPType>(), nNodesNewLevel * (TreeLevel::_nNodeImpProps + _nClasses), &status);
                DAAL_CHECK_STATUS_VAR(status);

                DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.doNodesSplit(nodeList, nNodes, nodeListNewLevel));

                levelNodeLists.push_back(nodeListNewLevel);
                levelNodeImpLists.push_back(impListNewLevel);

                DAAL_CHECK_STATUS_VAR(_treeLevelBuildHelper.doLevelPartition(indexedFeatures.getFullData(), nodeList, nNodes, treeOrderLev,
                                                                             treeOrderLevBuf, nSelectedRows, nFeatures));
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
            DAAL_CHECK_MALLOC(binValuesHost[i].get());
            binValues[i] = binValuesHost[i].get();
        }

        typename DFTreeConverterType::TreeHelperType mTreeHelper;

        DFTreeConverterType converter;
        DAAL_CHECK_STATUS_VAR(converter.convertToDFDecisionTree(DFTreeRecords, binValues.data(), mTreeHelper, _nClasses));

        mdImpl.add(mTreeHelper._tree, _nClasses);

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

    if (mdiRequired || mdaRequired) DAAL_CHECK_STATUS_VAR(varImpResPtr->releaseBlockOfRows(varImpBlock));

    DAAL_CHECK_STATUS_VAR(const_cast<NumericTable *>(x)->releaseBlockOfRows(dataBlock));

    return status;
}

} /* namespace internal */
} /* namespace training */
} /* namespace classification */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
