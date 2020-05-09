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
#include "service/kernel/oneapi/reducer.h"
#include "service/kernel/oneapi/select_indexed.h"
#include "service/kernel/oneapi/sorter.h"
#include "services/daal_defines.h"

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

class Range
{
public:
    static Range createFromBlock(uint32_t blockIndex, uint32_t maxBlockSize, uint32_t sumOfBlocksSize)
    {
        // TODO: check that arguments are correct

        const uint32_t startIndex = blockIndex * maxBlockSize;
        const uint32_t endIndex   = startIndex + maxBlockSize;
        return Range { startIndex, endIndex > sumOfBlocksSize ? sumOfBlocksSize : endIndex };
    }

    uint32_t startIndex;
    uint32_t endIndex;
    uint32_t count;

private:
    Range(uint32_t startIndex, uint32_t endIndex) : startIndex(startIndex), endIndex(endIndex), count(endIndex - startIndex) {}
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
    const uint32_t k                  = parameter->k;

    const size_t nQueryRows  = ntData->getNumberOfRows();
    const size_t nLabelRows  = labels->getNumberOfRows();
    const size_t nDataRows   = points->getNumberOfRows() < nLabelRows ? points->getNumberOfRows() : nLabelRows;
    const uint32_t nFeatures = points->getNumberOfColumns();

    // Block dimensions below are optimal for GEN9
    // Number of doubles is to 2X less against floats
    // to keep the same block size in bytes
    const uint32_t maxDataBlockRowCount  = 4096 * 4;
    const uint32_t maxQueryBlockRowCount = (2048 * 4) / sizeof(algorithmFpType);

    // Maximal number of partial selections to be merged at once
    const uint32_t selectionMaxNumberOfChunks = 16;
    const uint32_t histogramSize              = 256;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, maxDataBlockRowCount, maxQueryBlockRowCount);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, maxQueryBlockRowCount, k);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, maxQueryBlockRowCount * k, selectionMaxNumberOfChunks);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(uint32_t, maxQueryBlockRowCount, histogramSize);

    auto dataSumOfSquares = context.allocate(TypeIds::id<algorithmFpType>(), maxDataBlockRowCount, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto distances = context.allocate(TypeIds::id<algorithmFpType>(), maxDataBlockRowCount * maxQueryBlockRowCount, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto partialDistances = context.allocate(TypeIds::id<algorithmFpType>(), maxQueryBlockRowCount * k * selectionMaxNumberOfChunks, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto partialLabels = context.allocate(TypeIds::id<int>(), maxQueryBlockRowCount * k * selectionMaxNumberOfChunks, &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto sortedLabels = context.allocate(TypeIds::id<int>(), maxQueryBlockRowCount * k, &st);
    DAAL_CHECK_STATUS_VAR(st);
    // temporary buffer for RADIX sort
    auto radixBuffer = context.allocate(TypeIds::id<int>(), maxQueryBlockRowCount * histogramSize, &st);
    DAAL_CHECK_STATUS_VAR(st);

    const uint32_t nDataBlockCount      = nDataRows / maxDataBlockRowCount + uint32_t(nDataRows % maxDataBlockRowCount != 0);
    const uint32_t nQueryBlockCount     = nQueryRows / maxQueryBlockRowCount + uint32_t(nQueryRows % maxQueryBlockRowCount != 0);
    const uint32_t nSelectionBlockCount = nDataBlockCount / selectionMaxNumberOfChunks + uint32_t(nDataBlockCount % selectionMaxNumberOfChunks != 0);
    SelectIndexed::Result selectResult(context, k, maxQueryBlockRowCount, distances.type(), &st);
    DAAL_CHECK_STATUS_VAR(st);

    SelectIndexed::Params params(k, TypeIds::id<algorithmFpType>(), maxDataBlockRowCount, parameter->engine);
    SelectIndexedFactory factory;
    SharedPtr<SelectIndexed> selector(factory.create(k, params, &st));
    DAAL_CHECK_STATUS_VAR(st);

    for (uint32_t qblock = 0; qblock < nQueryBlockCount; qblock++)
    {
        Range curQueryRange = Range::createFromBlock(qblock, maxQueryBlockRowCount, nQueryRows);
        BlockDescriptor<algorithmFpType> queryRows;
        DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(curQueryRange.startIndex, curQueryRange.count, readOnly, queryRows));
        auto curQuery = queryRows.getBuffer();
        for (uint32_t sblock = 0; sblock < nSelectionBlockCount; sblock++)
        {
            uint32_t curSelectionMaxNumberOfChunks = sblock == 0 ? selectionMaxNumberOfChunks : selectionMaxNumberOfChunks - 1;
            uint32_t selectionChunkCount           = uint32_t(sblock != 0);
            Range curDataBlockRange                = Range::createFromBlock(sblock, curSelectionMaxNumberOfChunks, nDataBlockCount);
            for (uint32_t dblock = curDataBlockRange.startIndex; dblock < curDataBlockRange.endIndex; dblock++)
            {
                Range curDataRange = Range::createFromBlock(dblock, maxDataBlockRowCount, nDataRows);
                BlockDescriptor<int> labelRows;
                DAAL_CHECK_STATUS_VAR(labels->getBlockOfRows(curDataRange.startIndex, curDataRange.count, readOnly, labelRows));
                BlockDescriptor<algorithmFpType> dataRows;
                DAAL_CHECK_STATUS_VAR(points->getBlockOfRows(curDataRange.startIndex, curDataRange.count, readOnly, dataRows));
                // Collect sums of squares from train data
                auto sumResult = math::SumReducer::sum(math::Layout::RowMajor, dataRows.getBuffer(), curDataRange.count, nFeatures, &st);
                DAAL_CHECK_STATUS_VAR(st);
                // Initialize GEMM distances
                DAAL_CHECK_STATUS_VAR(scatterSumOfSquares(context, sumResult.sumOfSquares, curDataRange.count, curQueryRange.count, distances));
                // Let's calculate distances using GEMM
                DAAL_CHECK_STATUS_VAR(
                    computeDistances(context, dataRows.getBuffer(), curQuery, distances, curDataRange.count, curQueryRange.count, nFeatures));
                // Select k smallest distances and their labels from every row of the [curQueryRange.count]x[curDataRange.count] block
                selector->selectNearestDistancesAndLabels(distances, labelRows.getBuffer(), k, curQueryRange.count, curDataRange.count,
                                                          curDataRange.count, 0, selectResult, &st);
                DAAL_CHECK_STATUS_VAR(st);
                // copy block results to buffer in order to get merged with the same selection algorithm (up to selectionMaxNumberOfChunks of partial results)
                // and keep the first part containing previously merged result if exists
                DAAL_CHECK_STATUS_VAR(copyPartialDistancesAndLabels(context, selectResult.values, selectResult.indices, partialDistances,
                                                                    partialLabels, curQueryRange.count, k, selectionChunkCount,
                                                                    selectionMaxNumberOfChunks));
                DAAL_CHECK_STATUS_VAR(labels->releaseBlockOfRows(labelRows));
                DAAL_CHECK_STATUS_VAR(points->releaseBlockOfRows(dataRows));
                selectionChunkCount++;
            }
            // merge partial data by one more K-selection
            selector->selectNearestDistancesAndLabels(partialDistances, partialLabels, k, curQueryRange.count, k * curDataBlockRange.count,
                                                      k * selectionMaxNumberOfChunks, k * selectionMaxNumberOfChunks, selectResult, &st);
        }
        DAAL_CHECK_STATUS_VAR(st);
        // sort labels of closest neighbors
        RadixSort::sort(selectResult.indices, sortedLabels, radixBuffer, curQueryRange.count, k, k, &st);
        DAAL_CHECK_STATUS_VAR(st);
        BlockDescriptor<algorithmFpType> labelsBlock;
        DAAL_CHECK_STATUS_VAR(y->getBlockOfRows(curQueryRange.startIndex, curQueryRange.count, writeOnly, labelsBlock));
        // search for maximum occurrence label
        DAAL_CHECK_STATUS_VAR(computeWinners(context, sortedLabels, curQueryRange.count, k, labelsBlock.getBuffer()));
        DAAL_CHECK_STATUS_VAR(y->releaseBlockOfRows(labelsBlock));
        DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(queryRows));
    }
    return st;
}
template <typename algorithmFpType>
Status KNNClassificationPredictKernelUCAPI<algorithmFpType>::copyPartialDistancesAndLabels(
    ExecutionContextIface & context, const UniversalBuffer & distances, const UniversalBuffer & labels, UniversalBuffer & partialDistances,
    UniversalBuffer & partialLabels, uint32_t queryBlockRows, uint32_t k, uint32_t nChunk, uint32_t totalNumberOfChunks)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.copyPartialSelections);
    if (k > static_cast<uint32_t>(INT_MAX) || totalNumberOfChunks > static_cast<uint32_t>(INT_MAX) || nChunk > static_cast<uint32_t>(INT_MAX))
    {
        return services::Status(services::ErrorBufferSizeIntegerOverflow);
    }

    Status st;
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel_gather_selection = kernel_factory.getKernel("copy_partial_selection", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(7);
    args.set(0, distances, AccessModeIds::read);
    args.set(1, labels, AccessModeIds::read);
    args.set(2, partialDistances, AccessModeIds::readwrite);
    args.set(3, partialLabels, AccessModeIds::readwrite);
    args.set(4, k);
    args.set(5, nChunk);
    args.set(6, totalNumberOfChunks);

    KernelRange local_range(1, 1);
    KernelRange global_range(queryBlockRows, k);

    KernelNDRange range(2);
    range.global(global_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    context.run(range, kernel_gather_selection, args, &st);
    return st;
}

template <typename algorithmFpType>
Status KNNClassificationPredictKernelUCAPI<algorithmFpType>::scatterSumOfSquares(ExecutionContextIface & context,
                                                                                 const UniversalBuffer & dataSumOfSquares, uint32_t dataBlockRowCount,
                                                                                 uint32_t queryBlockRowCount, UniversalBuffer & distances)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.scatterSumOfSquares);
    if (dataBlockRowCount > static_cast<uint32_t>(INT_MAX))
    {
        return services::Status(services::ErrorBufferSizeIntegerOverflow);
    }
    Status st;
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel_init_distances = kernel_factory.getKernel("scatter_row", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(3);
    args.set(0, dataSumOfSquares, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::write);
    args.set(2, dataBlockRowCount);

    KernelRange global_range(dataBlockRowCount, queryBlockRowCount);
    context.run(global_range, kernel_init_distances, args, &st);
    return st;
}

template <typename algorithmFpType>
Status KNNClassificationPredictKernelUCAPI<algorithmFpType>::computeDistances(ExecutionContextIface & context, const Buffer<algorithmFpType> & data,
                                                                              const Buffer<algorithmFpType> & query, UniversalBuffer & distances,
                                                                              uint32_t dataBlockRowCount, uint32_t queryBlockRowCount,
                                                                              uint32_t nFeatures)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.GEMM);
    return BlasGpu<algorithmFpType>::xgemm(math::Layout::RowMajor, math::Transpose::NoTrans, math::Transpose::Trans, queryBlockRowCount,
                                           dataBlockRowCount, nFeatures, algorithmFpType(-2.0), query, nFeatures, 0, data, nFeatures, 0,
                                           algorithmFpType(1.0), distances.get<algorithmFpType>(), dataBlockRowCount, 0);
}

template <typename algorithmFpType>
Status KNNClassificationPredictKernelUCAPI<algorithmFpType>::computeWinners(ExecutionContextIface & context, const UniversalBuffer & labels,
                                                                            uint32_t queryBlockRowCount, uint32_t k, UniversalBuffer labelsOut)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.computeWinners);
    if (k > static_cast<uint32_t>(INT_MAX))
    {
        return services::Status(services::ErrorBufferSizeIntegerOverflow);
    }
    Status st;
    auto & kernel_factory = context.getClKernelFactory();
    DAAL_CHECK_STATUS_VAR(buildProgram(kernel_factory));
    auto kernel_compute_winners = kernel_factory.getKernel("find_max_occurance", &st);
    DAAL_CHECK_STATUS_VAR(st);

    KernelArguments args(3);
    args.set(0, labels, AccessModeIds::read);
    args.set(1, labelsOut, AccessModeIds::write);
    args.set(2, k);

    KernelRange local_range(1);
    KernelRange global_range(queryBlockRowCount);

    KernelNDRange range(1);
    range.global(global_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    range.local(local_range, &st);
    DAAL_CHECK_STATUS_VAR(st);
    context.run(range, kernel_compute_winners, args, &st);
    return st;
}

template <typename algorithmFpType>
Status KNNClassificationPredictKernelUCAPI<algorithmFpType>::buildProgram(ClKernelFactoryIface & kernel_factory)
{
    auto fptype_name   = oneapi::internal::getKeyFPType<algorithmFpType>();
    auto build_options = fptype_name;
    build_options.add(" -D sortedType=int -D NumParts=16 ");

    services::String cachekey("__daal_algorithms_bf_knn_block_");
    cachekey.add(fptype_name);

    Status st;
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), bf_knn_cl_kernels, build_options.c_str(), &st);
    }
    return st;
}

} // namespace internal
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
