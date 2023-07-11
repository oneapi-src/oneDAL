/* file: gbt_regression_train_dense_default_oneapi_impl.i */
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
//  Implementation of auxiliary functions for gradient boosted trees regression
//  (defaultDense) method.
//--
*/

#ifndef __GBT_REGRESSION_TRAIN_DENSE_DEFAULT_ONEAPI_IMPL_I__
#define __GBT_REGRESSION_TRAIN_DENSE_DEFAULT_ONEAPI_IMPL_I__

#include "src/algorithms/dtrees/gbt/regression/oneapi/cl_kernels/gbt_batch_regression_kernels.cl"

#include "src/algorithms/dtrees/gbt/oneapi/gbt_feature_type_helper_oneapi.i"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_model_impl.h"
#include "src/algorithms/dtrees/gbt/regression/gbt_regression_tree_impl.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"

#include "src/externals/service_profiler.h"
#include "services/internal/buffer.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "services/error_indexes.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_algo_utils.h"
#include "services/internal/sycl/types.h"

using namespace daal::algorithms::gbt::internal;
using namespace daal::algorithms::gbt::regression::internal;

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
template <typename algorithmFPType>
static services::Status buildProgram(ClKernelFactoryIface & factory)
{
    services::Status status;

    DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);

    auto fptype_name   = getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;
    build_options.add("-cl-std=CL1.2");

    services::String cachekey("__daal_algorithms_gbt_batch_regression_");
    cachekey.add(fptype_name);
    factory.build(ExecutionTargetIds::device, cachekey.c_str(), gbt_batch_regression_kernels, build_options.c_str(), status);

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::scan(const services::internal::Buffer<algorithmFPType> & values,
                                                                                 UniversalBuffer & partialSums, uint32_t nRows, uint32_t localSize,
                                                                                 uint32_t nLocalSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.scan);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelScan;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(values), algorithmFPType, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialSums, algorithmFPType, nLocalSums);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, values, AccessModeIds::read);
        args.set(1, partialSums, AccessModeIds::write);
        args.set(2, nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalSums);

        KernelNDRange range(1);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::reduce(UniversalBuffer & partialSums, UniversalBuffer & totalSum,
                                                                                   uint32_t localSize, uint32_t nSubgroupSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.reduce);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelReduce;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialSums, algorithmFPType, nSubgroupSums);
        DAAL_ASSERT_UNIVERSAL_BUFFER(totalSum, algorithmFPType, 1);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, partialSums, AccessModeIds::read);
        args.set(1, totalSum, AccessModeIds::write);
        args.set(2, nSubgroupSums);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::getInitialResponse(NumericTable & y, algorithmFPType * response)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.getInitialResponse);

    services::Status status;

    const uint32_t nRows = static_cast<uint32_t>(y.getNumberOfRows());

    auto & context = services::internal::getDefaultContext();

    const uint32_t subSize       = _preferableSubGroup;
    const uint32_t localSize     = _preferableSubGroup;
    const uint32_t nLocalSums    = 1024 * localSize < nRows ? 1024 : (nRows / localSize) + !!(nRows % localSize);
    const uint32_t nSubgroupSums = nLocalSums * (localSize / subSize);

    auto partialSums = context.allocate(TypeIds::id<algorithmFPType>(), nSubgroupSums, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto totalSum = context.allocate(TypeIds::id<algorithmFPType>(), 1, status);
    DAAL_CHECK_STATUS_VAR(status);

    BlockDescriptor<algorithmFPType> yBlock;
    DAAL_CHECK_STATUS_VAR(y.getBlockOfRows(0, nRows, readOnly, yBlock));
    auto yBuffer = yBlock.getBuffer();

    DAAL_CHECK_STATUS_VAR(scan(yBuffer, partialSums, nRows, localSize, nLocalSums));
    DAAL_CHECK_STATUS_VAR(reduce(partialSums, totalSum, localSize, nLocalSums));

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(totalSum, algorithmFPType, 1);
        auto totalSumHost = totalSum.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        DAAL_ASSERT(response);
        *response = totalSumHost.get()[0] / nRows;
    }

    DAAL_CHECK_STATUS_VAR(y.releaseBlockOfRows(yBlock));

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeOptCoeffs(NumericTable & y, UniversalBuffer & response,
                                                                                             UniversalBuffer & optCoeffs)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeOptCoeffs);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelComputeOptCoeffs;

    const uint32_t nRows = static_cast<uint32_t>(y.getNumberOfRows());

    BlockDescriptor<algorithmFPType> yBlock;
    DAAL_CHECK_STATUS_VAR(y.getBlockOfRows(0, nRows, readOnly, yBlock));
    auto yBuffer = yBlock.getBuffer();

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(UniversalBuffer(yBuffer), algorithmFPType, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(response, algorithmFPType, nRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(optCoeffs, algorithmFPType, nRows * 2);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, yBuffer, AccessModeIds::read);
        args.set(1, response, AccessModeIds::read);
        args.set(2, optCoeffs, AccessModeIds::write);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    DAAL_CHECK_STATUS_VAR(y.releaseBlockOfRows(yBlock));

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::initializeTreeOrder(uint32_t nRows, UniversalBuffer & treeOrder)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initializeTreeOrder);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelInitializeTreeOrder;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int, nRows);

        KernelArguments args(1, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, treeOrder, AccessModeIds::write);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computePartialHistograms(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & optCoeffs, UniversalBuffer & partialHistograms, uint32_t iStart,
    uint32_t nRows, UniversalBuffer & binOffsets, uint32_t nTotalBins, uint32_t nFeatures, uint32_t localSize, uint32_t nPartialHistograms,
    uint32_t totalRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computePartialHistograms);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelComputePartialHistograms;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(data, uint32_t, totalRows * nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int, totalRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(optCoeffs, algorithmFPType, totalRows * 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialHistograms, algorithmFPType, _maxLocalHistograms * nTotalBins * 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, nFeatures + 1);

        KernelArguments args(9, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, optCoeffs, AccessModeIds::read);
        args.set(3, partialHistograms, AccessModeIds::write);
        args.set(4, iStart);
        args.set(5, nRows);
        args.set(6, binOffsets, AccessModeIds::read);
        args.set(7, nTotalBins);
        args.set(8, nFeatures);

        uint32_t localSize = nFeatures < _maxLocalSize ? nFeatures : _maxLocalSize;

        KernelRange local_range(1, localSize);
        KernelRange global_range(nPartialHistograms, localSize);

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

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::reducePartialHistograms(UniversalBuffer & partialHistograms,
                                                                                                    UniversalBuffer & histograms, uint32_t nTotalBins,
                                                                                                    uint32_t reduceLocalSize,
                                                                                                    uint32_t nPartialHistograms)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.reducePartialHistograms);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelReducePartialHistograms;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialHistograms, algorithmFPType, _maxLocalHistograms * nTotalBins * 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(histograms, algorithmFPType, nTotalBins * 2);

        KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, partialHistograms, AccessModeIds::read);
        args.set(1, histograms, AccessModeIds::write);
        args.set(2, nPartialHistograms);
        args.set(3, nTotalBins);

        KernelRange local_range(1, reduceLocalSize);
        KernelRange global_range(nTotalBins, reduceLocalSize);

        KernelNDRange range(2);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeHistogram(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & optCoeffs, UniversalBuffer & partialHistograms,
    UniversalBuffer & histograms, uint32_t iStart, uint32_t nRows, UniversalBuffer & binOffsets, uint32_t nTotalBins, uint32_t totalRows,
    uint32_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeHistogram);

    services::Status status;

    const uint32_t localSize = _preferableSubGroup;
    const uint32_t nPartialHistograms =
        (nRows < _preferableGroupSize * _maxLocalHistograms) ? nRows / _preferableGroupSize + !!(nRows % _preferableGroupSize) : _maxLocalHistograms;

    uint32_t reduceLocalSize = 1;
    while (reduceLocalSize * 2 <= nPartialHistograms)
    {
        reduceLocalSize *= 2;
    }

    DAAL_CHECK_STATUS_VAR(computePartialHistograms(data, treeOrder, optCoeffs, partialHistograms, iStart, nRows, binOffsets, nTotalBins, nFeatures,
                                                   localSize, nPartialHistograms, totalRows));
    DAAL_CHECK_STATUS_VAR(reducePartialHistograms(partialHistograms, histograms, nTotalBins, reduceLocalSize, nPartialHistograms));

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeHistogramDiff(UniversalBuffer & histogramSrc,
                                                                                                 UniversalBuffer & histogramTotal,
                                                                                                 UniversalBuffer & histogramDst, uint32_t nTotalBins)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeHistogramDiff);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelComputeHistogramDiff;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(histogramSrc, algorithmFPType, nTotalBins * 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(histogramTotal, algorithmFPType, nTotalBins * 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(histogramDst, algorithmFPType, nTotalBins * 2);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, histogramSrc, AccessModeIds::read);
        args.set(1, histogramTotal, AccessModeIds::read);
        args.set(2, histogramDst, AccessModeIds::write);

        KernelRange global_range(nTotalBins);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeTotalOptCoeffs(UniversalBuffer & histograms,
                                                                                                  UniversalBuffer & totalOptCoeffs,
                                                                                                  UniversalBuffer & binOffsets, uint32_t nTotalBins,
                                                                                                  uint32_t nFeatures, uint32_t localSize)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeTotalOptCoeffs);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelComputeTotalOptCoeffs;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(histograms, algorithmFPType, nTotalBins * 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(totalOptCoeffs, algorithmFPType, 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, nFeatures + 1);

        KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, histograms, AccessModeIds::read);
        args.set(1, totalOptCoeffs, AccessModeIds::write);
        args.set(2, binOffsets, AccessModeIds::read);
        args.set(3, nTotalBins);

        KernelRange global_range(localSize, nFeatures);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeBestSplitForFeatures(
    UniversalBuffer & histograms, UniversalBuffer & totalOptCoeffs, UniversalBuffer & splitInfo, UniversalBuffer & splitValue,
    UniversalBuffer & binOffsets, uint32_t nTotalBins, uint32_t nFeatures, algorithmFPType lambda, uint32_t localSize)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeBestSplitForFeatures);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelComputeBestSplitForFeatures;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(histograms, algorithmFPType, nTotalBins * 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(totalOptCoeffs, algorithmFPType, 2);
        DAAL_ASSERT_UNIVERSAL_BUFFER(splitInfo, algorithmFPType, nFeatures * 5);
        DAAL_ASSERT_UNIVERSAL_BUFFER(splitValue, int, nFeatures);
        DAAL_ASSERT_UNIVERSAL_BUFFER(binOffsets, uint32_t, nFeatures + 1);

        KernelArguments args(7, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, histograms, AccessModeIds::read);
        args.set(1, totalOptCoeffs, AccessModeIds::read);
        args.set(2, splitInfo, AccessModeIds::write);
        args.set(3, splitValue, AccessModeIds::write);
        args.set(4, binOffsets, AccessModeIds::read);
        args.set(5, nTotalBins);
        args.set(6, lambda);

        KernelRange local_range(localSize, 1);
        KernelRange global_range(localSize, nFeatures);

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

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::computeBestSplit(UniversalBuffer & histograms,
                                                                                             UniversalBuffer & binOffsets, uint32_t nTotalBins,
                                                                                             uint32_t nFeatures, algorithmFPType lambda,
                                                                                             BestSplitOneAPI<algorithmFPType> & bestSplit,
                                                                                             algorithmFPType * gTotal, algorithmFPType * hTotal)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeBestSplit);

    services::Status status;

    auto & context      = services::internal::getDefaultContext();
    auto totalOptCoeffs = context.allocate(TypeIds::id<algorithmFPType>(), 2, status);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nFeatures, 5);
    auto splitInfo  = context.allocate(TypeIds::id<algorithmFPType>(), nFeatures * 5, status);
    auto splitValue = context.allocate(TypeIds::id<int>(), nFeatures, status);

    DAAL_CHECK_STATUS_VAR(status);

    const uint32_t localSize = _preferableSubGroup;

    DAAL_CHECK_STATUS_VAR(computeTotalOptCoeffs(histograms, totalOptCoeffs, binOffsets, nTotalBins, nFeatures, localSize));
    DAAL_CHECK_STATUS_VAR(
        computeBestSplitForFeatures(histograms, totalOptCoeffs, splitInfo, splitValue, binOffsets, nTotalBins, nFeatures, lambda, localSize));

    if (gTotal && hTotal)
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(totalOptCoeffs, algorithmFPType, 2);
        auto totalOptCoeffsHost = totalOptCoeffs.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        *gTotal = totalOptCoeffsHost.get()[0];
        *hTotal = totalOptCoeffsHost.get()[1];
    }
    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(splitInfo, algorithmFPType, nFeatures * 5);
        DAAL_ASSERT_UNIVERSAL_BUFFER(splitValue, int, nFeatures);
        auto splitInfoHost = splitInfo.template get<algorithmFPType>().toHost(ReadWriteMode::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        auto splitValueHost = splitValue.template get<int>().toHost(ReadWriteMode::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        for (uint32_t featId = 0; featId < nFeatures; featId++)
        {
            algorithmFPType impurityDecrease = splitInfoHost.get()[featId * 5 + 0];
            int32_t featureValue             = splitValueHost.get()[featId];
            if (featureValue != -1)
            {
                if (impurityDecrease > bestSplit._impurityDecrease
                    || (impurityDecrease == bestSplit._impurityDecrease && static_cast<int32_t>(featId) < bestSplit._featureIndex))
                {
                    bestSplit._impurityDecrease = impurityDecrease;
                    bestSplit._featureIndex     = static_cast<int32_t>(featId);
                    bestSplit._featureValue     = featureValue;
                    bestSplit._leftGTotal       = splitInfoHost.get()[featId * 5 + 1];
                    bestSplit._leftHTotal       = splitInfoHost.get()[featId * 5 + 2];
                    bestSplit._rightGTotal      = splitInfoHost.get()[featId * 5 + 3];
                    bestSplit._rightHTotal      = splitInfoHost.get()[featId * 5 + 4];
                }
            }
        }
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::partitionScan(const UniversalBuffer & data, UniversalBuffer & treeOrder,
                                                                                          UniversalBuffer & partialSums, int splitValue,
                                                                                          uint32_t iStart, uint32_t nRows, uint32_t localSize,
                                                                                          uint32_t nLocalSums, uint32_t totalRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionScan);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelPartitionScan;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(data, uint32_t, totalRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int, totalRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialSums, int, nLocalSums + 1);

        KernelArguments args(6, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, partialSums, AccessModeIds::write);
        args.set(3, splitValue);
        args.set(4, iStart);
        args.set(5, nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalSums);

        KernelNDRange range(1);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::partitionSumScan(UniversalBuffer & partialSums,
                                                                                             UniversalBuffer & partialPrefixSums,
                                                                                             UniversalBuffer & totalSum, uint32_t localSize,
                                                                                             uint32_t nSubgroupSums)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionSumScan);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelPartitionSumScan;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialSums, int, nSubgroupSums + 1);
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialPrefixSums, int, nSubgroupSums + 1);
        DAAL_ASSERT_UNIVERSAL_BUFFER(totalSum, int, 1);

        KernelArguments args(4, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, partialSums, AccessModeIds::read);
        args.set(1, partialPrefixSums, AccessModeIds::write);
        args.set(2, totalSum, AccessModeIds::write);
        args.set(3, nSubgroupSums);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize);

        KernelNDRange range(1);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::partitionReorder(
    const UniversalBuffer & data, UniversalBuffer & treeOrder, UniversalBuffer & treeOrderBuf, UniversalBuffer & partialPrefixSums, int splitValue,
    uint32_t iStart, uint32_t nRows, uint32_t localSize, uint32_t nLocalSums, uint32_t totalRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionReorder);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelPartitionReorder;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(data, uint32_t, totalRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int, totalRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrderBuf, int, totalRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(partialPrefixSums, int, nLocalSums + 1);

        KernelArguments args(7, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, data, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::read);
        args.set(2, treeOrderBuf, AccessModeIds::write);
        args.set(3, partialPrefixSums, AccessModeIds::read);
        args.set(4, splitValue);
        args.set(5, iStart);
        args.set(6, nRows);

        KernelRange local_range(localSize);
        KernelRange global_range(localSize * nLocalSums);

        KernelNDRange range(1);
        range.global(global_range, status);
        DAAL_CHECK_STATUS_VAR(status);
        range.local(local_range, status);
        DAAL_CHECK_STATUS_VAR(status);

        context.run(range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::partitionCopy(UniversalBuffer & treeOrderBuf, UniversalBuffer & treeOrder,
                                                                                          uint32_t iStart, uint32_t nRows, uint32_t totalRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.partitionCopy);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelPartitionCopy;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int, totalRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrderBuf, int, totalRows);

        KernelArguments args(3, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, treeOrderBuf, AccessModeIds::read);
        args.set(1, treeOrder, AccessModeIds::write);
        args.set(2, iStart);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::doPartition(const UniversalBuffer & data, UniversalBuffer & treeOrder,
                                                                                        UniversalBuffer & treeOrderBuf, int splitValue,
                                                                                        uint32_t iStart, uint32_t nRows, uint32_t & nLeft,
                                                                                        uint32_t & nRight, uint32_t totalRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.doPartition);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    const int subSize       = _preferableSubGroup;
    const int localSize     = _preferableSubGroup;
    const int nLocalSums    = _maxLocalSums * localSize < nRows ? _maxLocalSums : (nRows / localSize) + !!(nRows % localSize);
    const int nSubgroupSums = nLocalSums * (localSize / subSize);

    DAAL_OVERFLOW_CHECK_BY_ADDING(uint32_t, nSubgroupSums, 1);
    auto partialSums = context.allocate(TypeIds::id<int>(), nSubgroupSums + 1, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto partialPrefixSums = context.allocate(TypeIds::id<int>(), nSubgroupSums + 1, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto totalSum = context.allocate(TypeIds::id<int>(), 1, status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS_VAR(partitionScan(data, treeOrder, partialSums, splitValue, iStart, nRows, localSize, nLocalSums, totalRows));
    DAAL_CHECK_STATUS_VAR(partitionSumScan(partialSums, partialPrefixSums, totalSum, localSize, nSubgroupSums));
    DAAL_CHECK_STATUS_VAR(
        partitionReorder(data, treeOrder, treeOrderBuf, partialPrefixSums, splitValue, iStart, nRows, localSize, nLocalSums, totalRows));
    DAAL_CHECK_STATUS_VAR(partitionCopy(treeOrderBuf, treeOrder, iStart, nRows, totalRows));

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(totalSum, int, 1);
        auto totalSumHost = totalSum.template get<int>().toHost(ReadWriteMode::readOnly, status);
        DAAL_CHECK_STATUS_VAR(status);
        nRight = totalSumHost.get()[0];
        nLeft  = nRows - totalSumHost.get()[0];
        if (nLeft == 0 || nRight == 0)
        {
            return status;
        }
    }

    return status;
}

template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::updateResponse(UniversalBuffer & treeOrder, UniversalBuffer & response,
                                                                                           uint32_t iStart, uint32_t nRows, algorithmFPType inc,
                                                                                           uint32_t totalRows)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.updateResponse);

    services::Status status;

    auto & context = services::internal::getDefaultContext();

    auto & kernel = kernelUpdateResponse;

    {
        DAAL_ASSERT_UNIVERSAL_BUFFER(treeOrder, int, totalRows);
        DAAL_ASSERT_UNIVERSAL_BUFFER(response, algorithmFPType, totalRows);

        KernelArguments args(5, status);
        DAAL_CHECK_STATUS_VAR(status);
        args.set(0, treeOrder, AccessModeIds::read);
        args.set(1, response, AccessModeIds::write);
        args.set(2, iStart);
        args.set(3, nRows);
        args.set(4, inc);

        KernelRange global_range(nRows);

        context.run(global_range, kernel, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

//////////////////////////////////////////////////////////////////////////////////////////
// RegressionTrainBatchKernelOneAPI
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, gbt::regression::training::Method method>
services::Status RegressionTrainBatchKernelOneAPI<algorithmFPType, method>::compute(HostAppIface * pHostApp, const NumericTable * x,
                                                                                    const NumericTable * y, gbt::regression::Model & m, Result & res,
                                                                                    const Parameter & par, engines::internal::BatchBaseImpl & engine)
{
    typedef TreeTableConnector<algorithmFPType> ConnectorType;

    if (x->getNumberOfRows() > static_cast<size_t>(UINT_MAX) || x->getNumberOfColumns() > static_cast<size_t>(UINT_MAX))
    {
        return Status(ErrorBufferSizeIntegerOverflow);
    }

    const uint32_t nRows            = static_cast<uint32_t>(x->getNumberOfRows());
    const uint32_t nFeatures        = static_cast<uint32_t>(x->getNumberOfColumns());
    const uint32_t nFeaturesPerNode = static_cast<uint32_t>(par.featuresPerNode ? par.featuresPerNode : nFeatures);
    const bool inexactWithHistMethod =
        !par.memorySavingMode && par.splitMethod == gbt::training::inexact && x->getNumberOfColumns() == nFeaturesPerNode;

    DAAL_ASSERT(inexactWithHistMethod);

    gbt::internal::ModelImpl & modelImpl = *static_cast<daal::algorithms::gbt::regression::internal::ModelImpl *>(&m);
    DAAL_CHECK_MALLOC(modelImpl.reserve(par.maxIterations));

    services::Status status;

    auto & context        = Environment::getInstance()->getDefaultExecutionContext();
    auto & kernel_factory = context.getClKernelFactory();

    status |= buildProgram<algorithmFPType>(kernel_factory);
    DAAL_CHECK_STATUS_VAR(status);

    kernelScan                        = kernel_factory.getKernel("scan", status);
    kernelReduce                      = kernel_factory.getKernel("reduce", status);
    kernelInitializeTreeOrder         = kernel_factory.getKernel("initializeTreeOrder", status);
    kernelComputePartialHistograms    = kernel_factory.getKernel("computePartialHistograms", status);
    kernelReducePartialHistograms     = kernel_factory.getKernel("reducePartialHistograms", status);
    kernelComputeHistogramDiff        = kernel_factory.getKernel("computeHistogramDiff", status);
    kernelComputeOptCoeffs            = kernel_factory.getKernel("computeOptCoeffs", status);
    kernelComputeTotalOptCoeffs       = kernel_factory.getKernel("computeTotalOptCoeffs", status);
    kernelComputeBestSplitForFeatures = kernel_factory.getKernel("computeBestSplitForFeatures", status);
    kernelPartitionScan               = kernel_factory.getKernel("partitionScan", status);
    kernelPartitionSumScan            = kernel_factory.getKernel("partitionSumScan", status);
    kernelPartitionReorder            = kernel_factory.getKernel("partitionReorder", status);
    kernelPartitionCopy               = kernel_factory.getKernel("partitionCopy", status);
    kernelUpdateResponse              = kernel_factory.getKernel("updateResponse", status);

    DAAL_CHECK_STATUS_VAR(status);

    gbt::internal::IndexedFeaturesOneAPI<algorithmFPType> indexedFeatures;
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK_MALLOC(featTypes.init(*x));

    BinParams prm(par.maxBins, par.minBinSize);
    DAAL_CHECK_STATUS(status, (indexedFeatures.init(*const_cast<NumericTable *>(x), &featTypes, &prm)));

    auto response = context.allocate(TypeIds::id<algorithmFPType>(), nRows, status);
    DAAL_CHECK_STATUS_VAR(status);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, nRows, 2);
    auto optCoeffs = context.allocate(TypeIds::id<algorithmFPType>(), nRows * 2, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto treeOrder = context.allocate(TypeIds::id<int>(), nRows, status);
    DAAL_CHECK_STATUS_VAR(status);
    auto treeOrderBuf = context.allocate(TypeIds::id<int>(), nRows, status);
    DAAL_CHECK_STATUS_VAR(status);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _maxLocalHistograms, indexedFeatures.totalBins());
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, _maxLocalHistograms * indexedFeatures.totalBins(), 2);
    auto partialHistograms = context.allocate(TypeIds::id<algorithmFPType>(), _maxLocalHistograms * indexedFeatures.totalBins() * 2, status);
    DAAL_CHECK_STATUS_VAR(status);

    algorithmFPType initResp = 0.0;
    DAAL_CHECK_STATUS(status, getInitialResponse(*const_cast<NumericTable *>(y), &initResp));

    context.fill(response, initResp, status);
    DAAL_CHECK_STATUS_VAR(status);

    AOSNumericTablePtr treeStructure = ConnectorType::createGBTree(par.maxTreeDepth, status);
    DAAL_CHECK_STATUS_VAR(status);
    ConnectorType connector(treeStructure.get());

    const uint32_t maxNodes = static_cast<uint32_t>(treeStructure->getNumberOfRows());

    Collection<TreeNodeStorage> treeNodeStorages(maxNodes);
    DAAL_CHECK_MALLOC(treeNodeStorages.data());

    for (uint32_t iter = 0; (iter < par.maxIterations) && !algorithms::internal::isCancelled(status, pHostApp); ++iter)
    {
        DAAL_CHECK_STATUS_VAR(computeOptCoeffs(*const_cast<NumericTable *>(y), response, optCoeffs));
        DAAL_CHECK_STATUS_VAR(initializeTreeOrder(nRows, treeOrder));

        TableRecord<algorithmFPType> * record = connector.get(0);

        record->level       = 0;
        record->nid         = 0;
        record->iStart      = 0;
        record->n           = nRows;
        record->nodeState   = ConnectorType::split;
        record->isFinalized = false;

        Collection<SplitRecord<algorithmFPType> > splits;
        Collection<SplitRecord<algorithmFPType> > leafs;

        SplitRecord<algorithmFPType> splitRecord(record);
        splits.push_back(splitRecord);

        for (size_t splitId = 0; splitId < splits.size(); splitId++)
        {
            SplitRecord<algorithmFPType> & split = splits[splitId];
            if (split.first && split.second)
            {
                TableRecord<algorithmFPType> * leftRecord  = split.first;
                TableRecord<algorithmFPType> * rightRecord = split.second;
                DAAL_ASSERT(leftRecord->nid > 0 && leftRecord->nid < static_cast<size_t>(UINT_MAX));
                const uint32_t parentId = (static_cast<uint32_t>(leftRecord->nid) - 1) / 2;
                DAAL_CHECK_STATUS_VAR(treeNodeStorages[leftRecord->nid].allocate(indexedFeatures));
                DAAL_CHECK_STATUS_VAR(treeNodeStorages[rightRecord->nid].allocate(indexedFeatures));
                BestSplitOneAPI<algorithmFPType> bestSplitLeft;
                BestSplitOneAPI<algorithmFPType> bestSplitRight;
                if (leftRecord->n < rightRecord->n)
                {
                    DAAL_CHECK_STATUS_VAR(computeHistogram(indexedFeatures.getFullData(), treeOrder, optCoeffs, partialHistograms,
                                                           treeNodeStorages[leftRecord->nid].getHistograms(), leftRecord->iStart, leftRecord->n,
                                                           indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nRows, nFeatures));
                    DAAL_CHECK_STATUS_VAR(computeHistogramDiff(treeNodeStorages[leftRecord->nid].getHistograms(),
                                                               treeNodeStorages[parentId].getHistograms(),
                                                               treeNodeStorages[rightRecord->nid].getHistograms(), indexedFeatures.totalBins()));
                }
                else
                {
                    DAAL_CHECK_STATUS_VAR(computeHistogram(indexedFeatures.getFullData(), treeOrder, optCoeffs, partialHistograms,
                                                           treeNodeStorages[rightRecord->nid].getHistograms(), rightRecord->iStart, rightRecord->n,
                                                           indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nRows, nFeatures));
                    DAAL_CHECK_STATUS_VAR(computeHistogramDiff(treeNodeStorages[rightRecord->nid].getHistograms(),
                                                               treeNodeStorages[parentId].getHistograms(),
                                                               treeNodeStorages[leftRecord->nid].getHistograms(), indexedFeatures.totalBins()));
                }

                DAAL_CHECK_STATUS_VAR(computeBestSplit(treeNodeStorages[leftRecord->nid].getHistograms(), indexedFeatures.binOffsets(),
                                                       indexedFeatures.totalBins(), nFeatures, par.lambda, bestSplitLeft));
                DAAL_CHECK_STATUS_VAR(computeBestSplit(treeNodeStorages[rightRecord->nid].getHistograms(), indexedFeatures.binOffsets(),
                                                       indexedFeatures.totalBins(), nFeatures, par.lambda, bestSplitRight));

                bestSplitLeft._impurityDecrease -= (leftRecord->gTotal / (leftRecord->hTotal + par.lambda)) * leftRecord->gTotal;
                if (bestSplitLeft._impurityDecrease < par.minSplitLoss || bestSplitLeft._featureIndex < 0 || bestSplitLeft._featureValue < 0)
                {
                    leftRecord->isFinalized = true;
                    leftRecord->nodeState   = ConnectorType::badSplit;
                }
                else
                {
                    uint32_t nLeft  = 0;
                    uint32_t nRight = 0;
                    DAAL_CHECK_STATUS_VAR(doPartition(indexedFeatures.getFeature(bestSplitLeft._featureIndex), treeOrder, treeOrderBuf,
                                                      bestSplitLeft._featureValue, leftRecord->iStart, leftRecord->n, nLeft, nRight, nRows));
                    if (nLeft == 0 || nRight == 0)
                    {
                        leftRecord->isFinalized = true;
                        leftRecord->nodeState   = ConnectorType::badSplit;
                    }
                    else
                    {
                        leftRecord->isFinalized  = true;
                        leftRecord->featureValue = bestSplitLeft._featureValue;
                        leftRecord->featureIdx   = bestSplitLeft._featureIndex;
                        connector.createNode(leftRecord->level + 1, leftRecord->nid * 2 + 1, nLeft, leftRecord->iStart, bestSplitLeft._leftGTotal,
                                             bestSplitLeft._leftHTotal, nLeft, par);
                        connector.createNode(leftRecord->level + 1, leftRecord->nid * 2 + 2, leftRecord->n - nLeft, leftRecord->iStart + nLeft,
                                             bestSplitLeft._rightGTotal, bestSplitLeft._rightHTotal, nRight, par);
                        connector.setSplitLevel(leftRecord->level + 1);
                        connector.getSplitNodesMerged(leftRecord->nid, splits, false);
                    }
                }

                bestSplitRight._impurityDecrease -= (rightRecord->gTotal / (rightRecord->hTotal + par.lambda)) * rightRecord->gTotal;
                if (bestSplitRight._impurityDecrease < par.minSplitLoss || bestSplitRight._featureIndex < 0 || bestSplitRight._featureValue < 0)
                {
                    rightRecord->isFinalized = true;
                    rightRecord->nodeState   = ConnectorType::badSplit;
                }
                else
                {
                    uint32_t nLeft  = 0;
                    uint32_t nRight = 0;
                    DAAL_CHECK_STATUS_VAR(doPartition(indexedFeatures.getFeature(bestSplitRight._featureIndex), treeOrder, treeOrderBuf,
                                                      bestSplitRight._featureValue, rightRecord->iStart, rightRecord->n, nLeft, nRight, nRows));
                    if (nLeft == 0 || nRight == 0)
                    {
                        rightRecord->isFinalized = true;
                        rightRecord->nodeState   = ConnectorType::badSplit;
                    }
                    else
                    {
                        rightRecord->isFinalized  = true;
                        rightRecord->featureValue = bestSplitRight._featureValue;
                        rightRecord->featureIdx   = bestSplitRight._featureIndex;
                        connector.createNode(rightRecord->level + 1, rightRecord->nid * 2 + 1, nLeft, rightRecord->iStart, bestSplitRight._leftGTotal,
                                             bestSplitRight._leftHTotal, nLeft, par);
                        connector.createNode(rightRecord->level + 1, rightRecord->nid * 2 + 2, rightRecord->n - nLeft, rightRecord->iStart + nLeft,
                                             bestSplitRight._rightGTotal, bestSplitRight._rightHTotal, nRight, par);
                        connector.setSplitLevel(rightRecord->level + 1);
                        connector.getSplitNodesMerged(rightRecord->nid, splits, false);
                    }
                }

                treeNodeStorages[parentId].clear();
            }
            else
            {
                TableRecord<algorithmFPType> * record = (split.first ? split.first : split.second);
                DAAL_CHECK_STATUS_VAR(treeNodeStorages[record->nid].allocate(indexedFeatures));
                BestSplitOneAPI<algorithmFPType> bestSplit;
                algorithmFPType gTotal = 0.0;
                algorithmFPType hTotal = 0.0;
                DAAL_CHECK_STATUS_VAR(computeHistogram(indexedFeatures.getFullData(), treeOrder, optCoeffs, partialHistograms,
                                                       treeNodeStorages[record->nid].getHistograms(), record->iStart, record->n,
                                                       indexedFeatures.binOffsets(), indexedFeatures.totalBins(), nRows, nFeatures));
                DAAL_CHECK_STATUS_VAR(computeBestSplit(treeNodeStorages[record->nid].getHistograms(), indexedFeatures.binOffsets(),
                                                       indexedFeatures.totalBins(), nFeatures, par.lambda, bestSplit, &gTotal, &hTotal));
                if (record->nid == 0)
                {
                    record->gTotal = gTotal;
                    record->hTotal = hTotal;
                    record->nTotal = record->n;
                }
                bestSplit._impurityDecrease -= (record->gTotal / (record->hTotal + par.lambda)) * record->gTotal;
                if (bestSplit._impurityDecrease < par.minSplitLoss || bestSplit._featureIndex < 0 || bestSplit._featureValue < 0)
                {
                    record->isFinalized = true;
                    record->nodeState   = ConnectorType::badSplit;
                }
                else
                {
                    uint32_t nLeft  = 0;
                    uint32_t nRight = 0;
                    DAAL_CHECK_STATUS_VAR(doPartition(indexedFeatures.getFeature(bestSplit._featureIndex), treeOrder, treeOrderBuf,
                                                      bestSplit._featureValue, record->iStart, record->n, nLeft, nRight, nRows));
                    if (nLeft == 0 || nRight == 0)
                    {
                        record->isFinalized = true;
                        record->nodeState   = ConnectorType::badSplit;
                    }
                    else
                    {
                        record->isFinalized  = true;
                        record->featureValue = bestSplit._featureValue;
                        record->featureIdx   = bestSplit._featureIndex;
                        connector.createNode(record->level + 1, record->nid * 2 + 1, nLeft, record->iStart, bestSplit._leftGTotal,
                                             bestSplit._leftHTotal, nLeft, par);
                        connector.createNode(record->level + 1, record->nid * 2 + 2, record->n - nLeft, record->iStart + nLeft,
                                             bestSplit._rightGTotal, bestSplit._rightHTotal, nRight, par);
                        connector.setSplitLevel(record->level + 1);
                        connector.getSplitNodesMerged(record->nid, splits, false);
                    }
                }
                if (record->nid > 0)
                {
                    DAAL_ASSERT(record->nid > 0 && record->nid < static_cast<size_t>(UINT_MAX));
                    uint32_t parentId = (static_cast<uint32_t>(record->nid) - 1) / 2;
                    treeNodeStorages[parentId].clear();
                }
            }
        }

        Collection<TableRecord<algorithmFPType> *> leaves;
        connector.getLeafNodes(0, leaves);
        DAAL_ASSERT(leaves.size() < static_cast<size_t>(UINT_MAX));
        uint32_t nLeaves = static_cast<uint32_t>(leaves.size());

        for (uint32_t leafId = 0; leafId < nLeaves; leafId++)
        {
            TableRecord<algorithmFPType> * node = leaves[leafId];
            DAAL_ASSERT(node);

            algorithmFPType resp = 0;

            algorithmFPType val = node->hTotal + par.lambda;
            if (val != 0.0)
            {
                val                       = -node->gTotal / val;
                const algorithmFPType inc = val * par.shrinkage;

                resp = inc;

                DAAL_CHECK_STATUS_VAR(updateResponse(treeOrder, response, node->iStart, node->n, inc, nRows));
            }

            node->response    = resp;
            node->isFinalized = 1;
        }

        services::Collection<SharedPtr<algorithmFPType> > binValuesHost(nFeatures);
        DAAL_CHECK_MALLOC(binValuesHost.data());
        services::Collection<algorithmFPType *> binValues(nFeatures);
        DAAL_CHECK_MALLOC(binValues.data());

        for (uint32_t i = 0; i < nFeatures; i++)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER(indexedFeatures.binBorders(i), algorithmFPType, par.maxBins);
            binValuesHost[i] = indexedFeatures.binBorders(i).template get<algorithmFPType>().toHost(ReadWriteMode::readOnly, status);
            DAAL_CHECK_STATUS_VAR(status);
            binValues[i] = binValuesHost[i].get();
        }

        size_t maxLevel = 0;
        connector.getMaxLevel(0, maxLevel);
        DAAL_ASSERT(maxLevel + 1 <= 63);
        DAAL_ASSERT(((size_t)1 << (maxLevel + 1)) > 0 && ((size_t)1 << (maxLevel + 1)) < static_cast<size_t>(UINT_MAX));
        const uint32_t nNodes        = ((size_t)1 << (maxLevel + 1)) - 1;
        const uint32_t nNodesPresent = connector.getNNodes(0);

        gbt::internal::GbtDecisionTree * pTbl = new gbt::internal::GbtDecisionTree(nNodes, maxLevel, nNodesPresent);
        DAAL_CHECK_MALLOC(pTbl);

        HomogenNumericTable<double> * pTblImp = new HomogenNumericTable<double>(1, nNodes, NumericTable::doAllocate);
        DAAL_CHECK_MALLOC(pTblImp);
        HomogenNumericTable<int> * pTblSmplCnt = new HomogenNumericTable<int>(1, nNodes, NumericTable::doAllocate);
        DAAL_CHECK_MALLOC(pTblSmplCnt);

        DAAL_CHECK_STATUS_VAR(connector.template convertToGbtDecisionTree<DAAL_BASE_CPU>(
            binValues.data(), nNodes, maxLevel, pTbl, pTblImp->getArray(), pTblSmplCnt->getArray(), initResp, par));
        modelImpl.add(pTbl, pTblImp, pTblSmplCnt);
        initResp = 0.0;
    }

    return services::Status();
}

} /* namespace internal */
} /* namespace training */
} /* namespace regression */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
