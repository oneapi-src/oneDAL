/* file: bf_knn_classification_predict_kernel_ucapi_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifndef __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_UCAPI_IMPL_I__
#define __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_UCAPI_IMPL_I__

#include "algorithms/engines/engine.h"
#include "service/kernel/oneapi/sum_reducer.h"
#include "service/kernel/oneapi/select_indexed.h"
#include "service/kernel/oneapi/sorter.h"

#include "algorithms/kernel/k_nearest_neighbors/oneapi/bf_knn_classification_predict_kernel_ucapi.h"
#include "algorithms/kernel/k_nearest_neighbors/oneapi/bf_knn_classification_model_ucapi_impl.h"

#include "service/kernel/oneapi/blas_gpu.h"
#include "algorithms/kernel/k_nearest_neighbors/oneapi/cl_kernels/bf_knn_cl_kernels.cl"

#include "externals/service_ittnotify.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace prediction
{
namespace internal
{
using namespace daal::oneapi::internal;
using namespace services;
using sort::RadixSort;
using selection::QuickSelectIndexed;
using selection::SelectIndexed;
using selection::SelectIndexedFactory;

inline void mergePartialSelection(bool forceMerge, UniversalBuffer & partialDistances, UniversalBuffer & partialLabels, uint32_t & partCount,
                                  uint32_t selectionNumberOfChunks, uint32_t queryBlockRows, uint32_t nK, const SharedPtr<SelectIndexed> selector,
                                  SelectIndexed::Result & selectResult, Status * st)
{
    if (partCount >= selectionNumberOfChunks || forceMerge && partCount > 0)
    {
        // Select nK closest neighbors from [queryBlockRows]x[nK * partCount] buffers with partial results
        selector->selectNearestDistancesAndLabels(partialDistances, partialLabels, nK, queryBlockRows, nK * partCount, nK * selectionNumberOfChunks,
                                                  nK * selectionNumberOfChunks, selectResult, st);
        partCount = 1;
    }
}

struct BlockOfRows
{
    BlockOfRows(uint32_t blockIndex, uint32_t maxSize, uint32_t nRowsTotal)
    {
        first         = blockIndex * maxSize;
        uint32_t last = first + maxSize;
        last          = last > nRowsTotal ? nRowsTotal : last;
        number        = last - first;
    }
    uint32_t first;
    uint32_t number;
};

template <typename algorithmFpType>
Status KNNClassificationPredictKernelUCAPI<algorithmFpType>::compute(const NumericTable * x, const classifier::Model * m, NumericTable * y,
                                                                     const daal::algorithms::Parameter * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    Status st;

    auto & context = Environment::getInstance()->getDefaultExecutionContext();

    const Model * model = static_cast<const Model *>(m);

    NumericTable * ntData = const_cast<NumericTable *>(x);
    NumericTable * points = const_cast<NumericTable *>(model->impl()->getData().get());
    NumericTable * labels = const_cast<NumericTable *>(model->impl()->getLabels().get());

    const Parameter * const parameter = static_cast<const Parameter *>(par);
    const uint32_t nK                 = parameter->k;

    const size_t nQueryRows  = ntData->getNumberOfRows();
    const size_t nLabelRows  = labels->getNumberOfRows();
    const size_t nDataRows   = points->getNumberOfRows() < nLabelRows ? points->getNumberOfRows() : nLabelRows;
    const uint32_t nFeatures = points->getNumberOfColumns();

    // Block dimensions below are optimal for GEN9
    // Number of doubles is to 2X less against floats
    // to keep the same block size in bytes
    const uint32_t maxDataBlockRows  = 4096 * 4;
    const uint32_t maxQueryBlockRows = (2048 * 4) / sizeof(algorithmFpType);

    // Maximal number of partial selections to be merged at once
    const uint32_t selectionNumberOfChunks = 16;
    const uint32_t histogramSize           = 256;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, maxDataBlockRows, maxQueryBlockRows);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, maxQueryBlockRows, nK);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, maxQueryBlockRows * nK, selectionNumberOfChunks);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, maxQueryBlockRows, histogramSize);

    auto dataSumOfSquares = context.allocate(TypeIds::id<algorithmFpType>(), maxDataBlockRows, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto distances = context.allocate(TypeIds::id<algorithmFpType>(), maxDataBlockRows * maxQueryBlockRows, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto partialDistances = context.allocate(TypeIds::id<algorithmFpType>(), maxQueryBlockRows * nK * selectionNumberOfChunks, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto partialLabels = context.allocate(TypeIds::id<int>(), maxQueryBlockRows * nK * selectionNumberOfChunks, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto sorted = context.allocate(TypeIds::id<int>(), maxQueryBlockRows * nK, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto radix_buffer = context.allocate(TypeIds::id<int>(), maxQueryBlockRows * histogramSize, &st);
    DAAL_CHECK_STATUS_VAR(st);

    const uint32_t nDataBlocks  = nDataRows / maxDataBlockRows + uint32_t(nDataRows % maxDataBlockRows != 0);
    const uint32_t nQueryBlocks = nQueryRows / maxQueryBlockRows + uint32_t(nQueryRows % maxQueryBlockRows != 0);
    SelectIndexed::Result selectResult(context, nK, maxQueryBlockRows, distances.type(), &st);
    DAAL_CHECK_STATUS_VAR(st);

    SelectIndexed::Params params(nK, TypeIds::id<algorithmFpType>(), maxDataBlockRows, parameter->engine);
    SelectIndexedFactory factory;
    SharedPtr<SelectIndexed> selector(factory.create(nK, params, &st));
    DAAL_CHECK_STATUS_VAR(st);

    for (uint32_t qblock = 0; qblock < nQueryBlocks; qblock++)
    {
        BlockOfRows curQueryRows(qblock, maxQueryBlockRows, nQueryRows);
        BlockDescriptor<algorithmFpType> queryRows;
        DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(curQueryRows.first, curQueryRows.number, readOnly, queryRows));
        auto curQuery = queryRows.getBuffer();

        uint32_t partCount = 0;
        for (uint32_t dblock = 0; dblock < nDataBlocks; dblock++)
        {
            BlockOfRows curDataRows(dblock, maxDataBlockRows, nDataRows);
            BlockDescriptor<int> labelRows;
            DAAL_CHECK_STATUS_VAR(labels->getBlockOfRows(curDataRows.first, curDataRows.number, readOnly, labelRows));
            auto dataLabels = labelRows.getBuffer();
            BlockDescriptor<algorithmFpType> dataRows;
            DAAL_CHECK_STATUS_VAR(points->getBlockOfRows(curDataRows.first, curDataRows.number, readOnly, dataRows));
            auto curData = dataRows.getBuffer();
            // Collect sums of squares from train data
            auto sumResult = math::SumReducer::sum(math::Layout::RowMajor, curData, curDataRows.number, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            // Initialize GEMM distances
            scatterSumOfSquares(context, sumResult.sumOfSquares, distances, curDataRows.number, curQueryRows.number, &st);
            DAAL_CHECK_STATUS_VAR(st);
            // Let's calculate distances using GEMM
            computeDistances(context, curData, curQuery, distances, curDataRows.number, curQueryRows.number, nFeatures, &st);
            DAAL_CHECK_STATUS_VAR(st);
            // Select nK smallest distances and their labels from every row of the [curQueryRows.number]x[curDataRows.number] block
            selector->selectNearestDistancesAndLabels(distances, dataLabels, nK, curQueryRows.number, curDataRows.number, curDataRows.number, 0,
                                                      selectResult, &st);
            DAAL_CHECK_STATUS_VAR(st);
            // copy block results to buffer in order to get merged with the same selection algorithm (up to selectionNumberOfChunks of partial results)
            // and keep the first part containing previously merged result if exists
            copyPartialSelections(context, selectResult.values, selectResult.indices, partialDistances, partialLabels, curQueryRows.number, nK,
                                  partCount, selectionNumberOfChunks, &st);
            DAAL_CHECK_STATUS_VAR(st);
            partCount++;
            //merge partial data
            mergePartialSelection(false, partialDistances, partialLabels, partCount, selectionNumberOfChunks, curQueryRows.number, nK, selector,
                                  selectResult, &st);
            DAAL_CHECK_STATUS_VAR(st);
            DAAL_CHECK_STATUS_VAR(labels->releaseBlockOfRows(labelRows));
            DAAL_CHECK_STATUS_VAR(points->releaseBlockOfRows(dataRows));
        }
        // force merging of remainig partial data
        mergePartialSelection(true, partialDistances, partialLabels, partCount, selectionNumberOfChunks, curQueryRows.number, nK, selector,
                              selectResult, &st);
        DAAL_CHECK_STATUS_VAR(st);
        // sort labels of closest neighbors
        RadixSort::sort(selectResult.indices, sorted, radix_buffer, curQueryRows.number, nK, nK, &st);
        DAAL_CHECK_STATUS_VAR(st);
        BlockDescriptor<int> resultBlock;
        DAAL_CHECK_STATUS_VAR(y->getBlockOfRows(curQueryRows.first, curQueryRows.number, writeOnly, resultBlock));
        auto classes = UniversalBuffer(resultBlock.getBuffer());
        // search for maximum occurrence label
        computeWinners(context, sorted, classes, curQueryRows.number, nK, &st);
        DAAL_CHECK_STATUS_VAR(st);
        DAAL_CHECK_STATUS_VAR(y->releaseBlockOfRows(resultBlock));
        DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(queryRows));
    }
    return st;
}
template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::copyPartialSelections(ExecutionContextIface & context, const UniversalBuffer & distances,
                                                                                 const UniversalBuffer & categories,
                                                                                 UniversalBuffer & partialDistances, UniversalBuffer & partialLabels,
                                                                                 uint32_t queryBlockRows, uint32_t nK, uint32_t nChunk,
                                                                                 uint32_t totalNumberOfChunks, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.copyPartialSelections);
    if (nK > INT_MAX || totalNumberOfChunks > INT_MAX || nChunk > INT_MAX)
    {
        if (st)
        {
            *st = services::Status(services::ErrorBufferSizeIntegerOverflow);
        }
        return;
    }

    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel_gather_selection = kernel_factory.getKernel("gather_partial_selection", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(7);
    args.set(0, distances, AccessModeIds::read);
    args.set(1, categories, AccessModeIds::read);
    args.set(2, partialDistances, AccessModeIds::readwrite);
    args.set(3, partialLabels, AccessModeIds::readwrite);
    args.set(4, nK);
    args.set(5, nChunk);
    args.set(6, totalNumberOfChunks);

    KernelRange local_range(1, 1);
    KernelRange global_range(queryBlockRows, nK);

    KernelNDRange range(2);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel_gather_selection, args, st);
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::scatterSumOfSquares(ExecutionContextIface & context,
                                                                               const UniversalBuffer & dataSumOfSquares, UniversalBuffer & distances,
                                                                               uint32_t dataBlockRows, uint32_t queryBlockRows, Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.scatterSumOfSquares);
    if (dataBlockRows > INT_MAX)
    {
        if (st)
        {
            *st = services::Status(services::ErrorBufferSizeIntegerOverflow);
        }
        return;
    }
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel_init_distances = kernel_factory.getKernel("scatter_row", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(3);
    args.set(0, dataSumOfSquares, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::write);
    args.set(2, dataBlockRows);

    KernelRange global_range(dataBlockRows, queryBlockRows);
    context.run(global_range, kernel_init_distances, args, st);
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::computeDistances(ExecutionContextIface & context, const Buffer<algorithmFpType> & data,
                                                                            const Buffer<algorithmFpType> & probes, UniversalBuffer & distances,
                                                                            uint32_t dataBlockRows, uint32_t queryBlockRows, uint32_t nFeatures,
                                                                            Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.GEMM);
    auto gemmStatus = BlasGpu<algorithmFpType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, queryBlockRows,
                                                      dataBlockRows, nFeatures, algorithmFpType(-2.0), probes, nFeatures, 0, data, nFeatures, 0,
                                                      algorithmFpType(1.0), distances.get<algorithmFpType>(), dataBlockRows, 0);
    if (st != nullptr)
    {
        *st = gemmStatus;
    }
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::computeWinners(ExecutionContextIface & context, const UniversalBuffer & categories,
                                                                          UniversalBuffer & classes, uint32_t queryBlockRows, uint32_t nK,
                                                                          Status * st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeWinners);
    if (nK > INT_MAX)
    {
        if (st)
        {
            *st = services::Status(services::ErrorBufferSizeIntegerOverflow);
        }
        return;
    }
    auto & kernel_factory = context.getClKernelFactory();
    buildProgram(kernel_factory, st);
    DAAL_CHECK_STATUS_PTR(st);
    auto kernel_compute_winners = kernel_factory.getKernel("find_max_occurance", st);
    DAAL_CHECK_STATUS_PTR(st);

    KernelArguments args(3);
    args.set(0, categories, AccessModeIds::read);
    args.set(1, classes, AccessModeIds::write);
    args.set(2, nK);

    KernelRange local_range(1);
    KernelRange global_range(queryBlockRows);

    KernelNDRange range(1);
    range.global(global_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st);
    DAAL_CHECK_STATUS_PTR(st);
    context.run(range, kernel_compute_winners, args, st);
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::buildProgram(ClKernelFactoryIface & kernel_factory, Status * st)
{
    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFpType>();
    auto build_options = fptype_name;
    build_options.add(" -D sortedType=int -D NumParts=16 ");

    services::String cachekey("__daal_algorithms_bf_knn_block_");
    cachekey.add(fptype_name);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), bf_knn_cl_kernels, build_options.c_str(), st);
        DAAL_CHECK_STATUS_PTR(st);
    }
}

} // namespace internal
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
