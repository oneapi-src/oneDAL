/* file: bf_knn_classification_predict_kernel_ucapi_impl.i */
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

#ifndef __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_UCAPI_IMPL_I__
#define __BF_KNN_CLASSIFICATION_PREDICT_KERNEL_UCAPI_IMPL_I__

#include "service_rng.h"

#include "engine_batch_impl.h"

#include "oneapi/sum_reducer.h"
#include "oneapi/select_indexed.h"
#include "oneapi/sorter.h"

#include "numeric_table.h"
#include "oneapi/bf_knn_classification_predict_kernel_ucapi.h"
#include "oneapi/bf_knn_classification_model_ucapi_impl.h"

#include "oneapi/service_defines_oneapi.h"
#include "oneapi/blas_gpu.h"
#include "cl_kernels/bf_knn_cl_kernels.cl"

#include "service_ittnotify.h"

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

//using namespace daal::services::internal;
//using namespace daal::services;
//using namespace daal::internal;
using namespace daal::oneapi::internal;
using sort::RadixSort;
using selection::QuickSelectIndexed;

template<typename algorithmFpType>
Status KNNClassificationPredictKernelUCAPI<algorithmFpType>::
                 compute(const NumericTable * x, const classifier::Model * m, NumericTable * y, const daal::algorithms::Parameter * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute);

    Status st;

    auto& context = Environment::getInstance()->getDefaultExecutionContext();
    auto& kernel_factory = context.getClKernelFactory();

    const Model *model = static_cast<const Model *>(m);

    NumericTable *ntData = const_cast<NumericTable *>( x );
    NumericTable *points = const_cast<NumericTable *>(model->impl()->getData().get());
    NumericTable *labels = const_cast<NumericTable *>(model->impl()->getLabels().get());

    const Parameter * const parameter = static_cast<const Parameter *>(par);
    const auto nK = parameter->k;

    const size_t nProbeRows = ntData->getNumberOfRows();
    const size_t nLabelRows = labels->getNumberOfRows();
    const size_t nDataRows = points->getNumberOfRows() < nLabelRows ? points->getNumberOfRows() : nLabelRows;
    const size_t nFeatures = points->getNumberOfColumns();

    auto fptype_name = oneapi::internal::getKeyFPType<algorithmFpType>();
    auto build_options = fptype_name;
    build_options.add(" -D sortedType=int -D NumParts=16 ");

    services::String cachekey("__daal_algorithms_bf_knn_block_");
    cachekey.add(fptype_name);
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.buildProgram);
        kernel_factory.build(ExecutionTargetIds::device, cachekey.c_str(), bf_knn_cl_kernels, build_options.c_str());
    }

    const size_t dataBlockSize = 4096 * 4;
    const size_t probeBlockSize = (2048 * 4) / sizeof(algorithmFpType);
    const size_t nParts = 16;

    auto dataSq             = context.allocate(TypeIds::id<algorithmFpType>(), dataBlockSize,                       &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto data_temp          = context.allocate(TypeIds::id<algorithmFpType>(), dataBlockSize * nFeatures,           &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto probe_temp         = context.allocate(TypeIds::id<algorithmFpType>(), probeBlockSize * nFeatures,          &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto distances          = context.allocate(TypeIds::id<algorithmFpType>(), dataBlockSize * probeBlockSize,      &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto categories         = context.allocate(TypeIds::id<int>(), probeBlockSize * dataBlockSize,                  &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto partialDistances   = context.allocate(TypeIds::id<algorithmFpType>(), probeBlockSize * nK * (nParts + 1),  &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto partialCategories  = context.allocate(TypeIds::id<int>(), probeBlockSize * nK * (nParts + 1),              &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto sorted             = context.allocate(TypeIds::id<int>(), probeBlockSize * nK,                             &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto buffer             = context.allocate(TypeIds::id<int>(), probeBlockSize * nK,                             &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto radix_buffer       = context.allocate(TypeIds::id<int>(), probeBlockSize * 256,                            &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto classes            = context.allocate(TypeIds::id<int>(), probeBlockSize,                                  &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto rndSeq             = context.allocate(TypeIds::id<algorithmFpType>(), dataBlockSize,                       &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto rndIndices         = context.allocate(TypeIds::id<size_t>(), dataBlockSize,                                &st);
    DAAL_CHECK_STATUS_VAR(st);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(compute.RNG);
        auto engineImpl = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*parameter->engine));
        if(!engineImpl)
            return Status(ErrorIncorrectEngineParameter);
        daal::internal::RNGs<size_t, sse2> rng;
        size_t              numbers[dataBlockSize];
        algorithmFpType     values[dataBlockSize];
        rng.uniform(dataBlockSize, &numbers[0], engineImpl->getState(), 0, (size_t)(dataBlockSize - 1));
        for(int i = 0; i < dataBlockSize; i++)
            values[i] = static_cast<algorithmFpType>(numbers[i]) / (dataBlockSize - 1);
        context.copy(rndSeq, 0, (void*)&values[0], 0, dataBlockSize, &st);
        DAAL_CHECK_STATUS_VAR(st);
    }
    auto init_distances = kernel_factory.getKernel("init_distances", &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto find_winners = kernel_factory.getKernel("find_max_occurance", &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto gather_selection = kernel_factory.getKernel("gather_partial_selection", &st);
    DAAL_CHECK_STATUS_VAR(st);
    auto init_categories = kernel_factory.getKernel("init_categories", &st);
    DAAL_CHECK_STATUS_VAR(st);

    const size_t nDataBlocks = nDataRows / dataBlockSize + int (nDataRows % dataBlockSize != 0);
    const size_t nProbeBlocks = nProbeRows / probeBlockSize + int (nProbeRows % probeBlockSize != 0);
    QuickSelectIndexed::Result selectResult(context, nK, probeBlockSize, distances.type(), categories.type(), &st);
    DAAL_CHECK_STATUS_VAR(st);

    BlockDescriptor<algorithmFpType> probeRows;
    DAAL_CHECK_STATUS_VAR(ntData->getBlockOfRows(0, nProbeRows, readOnly, probeRows));
    auto probes_all = probeRows.getBuffer();

    BlockDescriptor<algorithmFpType> dataRows;
    DAAL_CHECK_STATUS_VAR(points->getBlockOfRows(0, nDataRows, readOnly, dataRows));
    auto data_all = dataRows.getBuffer();

    BlockDescriptor<int> labelRows;
    DAAL_CHECK_STATUS_VAR(labels->getBlockOfRows(0, nDataRows, readOnly, labelRows));
    auto labels_all = labelRows.getBuffer();

    for (size_t pblock = 0; pblock < nProbeBlocks; pblock++)
    {
        const size_t pfirst = pblock * probeBlockSize;
        size_t plast = pfirst + probeBlockSize;

        if (plast > nProbeRows)
            plast = nProbeRows;
        const size_t curProbeBlockSize = plast - pfirst;
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.copy_query);
            context.copy(probe_temp, 0, probes_all, pfirst * nFeatures, nFeatures * curProbeBlockSize, &st);
        }
        auto probes = probe_temp.template get<algorithmFpType>();
        uint32_t partCount = 0;

        for (size_t dblock = 0; dblock < nDataBlocks; dblock++)
        {
            const size_t dfirst = dblock * dataBlockSize;
            size_t dlast = dfirst + dataBlockSize;

            if (dlast > nDataRows)
                dlast = nDataRows;
            const size_t curDataBlockSize = dlast - dfirst;

            // Let's calculate distances using GEMM
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(compute.copy_data);
                context.copy(data_temp, 0, data_all, dfirst * nFeatures, nFeatures * curDataBlockSize, &st);
            }
            auto data = data_temp.template get<algorithmFpType>();
            auto sumResult = math::SumReducer::sum(math::Layout::RowMajor,
                                                   data, curDataBlockSize, nFeatures, &st); DAAL_CHECK_STATUS_VAR(st);
            initDistances(context, init_distances, sumResult.sumOfSquares, distances, curDataBlockSize, curProbeBlockSize, &st); DAAL_CHECK_STATUS_VAR(st);
            computeDistances(context, data, probes, distances, curDataBlockSize, curProbeBlockSize, nFeatures, &st); DAAL_CHECK_STATUS_VAR(st);
            {
                DAAL_ITTNOTIFY_SCOPED_TASK(compute.copy_categories);
                initCategories(context, init_categories, labels_all, categories, curProbeBlockSize, curDataBlockSize, dfirst, &st); DAAL_CHECK_STATUS_VAR(st);
            }
            QuickSelectIndexed::select( distances, categories, rndSeq, dataBlockSize, nK, curProbeBlockSize,
                                        curDataBlockSize, curDataBlockSize, selectResult, &st);  DAAL_CHECK_STATUS_VAR(st);
            copyPartialSelections(context, gather_selection, selectResult.values, selectResult.indices, partialDistances,
                                  partialCategories, curProbeBlockSize, nK, partCount, nParts, dblock != 0 && partCount == 0, &st);  DAAL_CHECK_STATUS_VAR(st);
            partCount++;
            if(partCount >= nParts)
            {
                QuickSelectIndexed::select(partialDistances, partialCategories, rndSeq, dataBlockSize, nK, curProbeBlockSize,
                                            nK * partCount, nK * nParts, selectResult, &st);  DAAL_CHECK_STATUS_VAR(st);
                partCount = 0;
            }
        }
        if(partCount > 0)
        {
            QuickSelectIndexed::select(partialDistances, partialCategories, rndSeq, dataBlockSize, nK, curProbeBlockSize,
                                        nK * partCount, nK * nParts, selectResult, &st);  DAAL_CHECK_STATUS_VAR(st);
        }
        RadixSort::sort(selectResult.indices, sorted,  radix_buffer, curProbeBlockSize, nK, nK, &st); DAAL_CHECK_STATUS_VAR(st);
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.winners);
            computeWinners(context, find_winners, sorted, classes, curProbeBlockSize, nK, &st); DAAL_CHECK_STATUS_VAR(st);
        }
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(compute.copy_results);
            BlockDescriptor<algorithmFpType> resultBlock;
            DAAL_CHECK_STATUS_VAR(y->getBlockOfRows(0, nProbeRows, writeOnly, resultBlock));
            auto classesRows = classes.template get<int>().toHost(ReadWriteMode::readOnly);
            auto res = resultBlock.getBuffer().toHost(ReadWriteMode::writeOnly);
                for(int i = 0; i < curProbeBlockSize; i++)
                {
                    res.get()[pfirst + i] = classesRows.get()[i];
                }
            DAAL_CHECK_STATUS_VAR(y->releaseBlockOfRows(resultBlock));
        }
    }
    DAAL_CHECK_STATUS_VAR(ntData->releaseBlockOfRows(probeRows));
    DAAL_CHECK_STATUS_VAR(points->releaseBlockOfRows(dataRows));
    DAAL_CHECK_STATUS_VAR(labels->releaseBlockOfRows(labelRows));
    return st;
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::initCategories
        (ExecutionContextIface& context,
         const KernelPtr& kernel_init_categories,
         const Buffer<int>& labels,
         UniversalBuffer& categories,
         uint32_t probeBlockSize,
         uint32_t dataBlockSize,
         uint32_t offset,
         Status* st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.init_categories);
    KernelArguments args(4);
    args.set(0, labels, AccessModeIds::read);
    args.set(1, categories, AccessModeIds::write);
    args.set(2, dataBlockSize);
    args.set(3, offset);

    KernelRange global_range(dataBlockSize, probeBlockSize);

    {
        context.run(global_range, kernel_init_categories, args, st);
    }
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::copyPartialSelections
        (ExecutionContextIface& context,
         const KernelPtr& kernel_gather_selection,
         UniversalBuffer& distances,
         UniversalBuffer& categories,
         UniversalBuffer& partialDistances,
         UniversalBuffer& partialCategories,
         uint32_t probeBlockSize,
         uint32_t nK,
         uint32_t& nPart,
         uint32_t totalParts,
         bool bKeepPrevious,
         Status* st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.copyPartialSelections);
    if(bKeepPrevious)
        nPart++;
    KernelArguments args(7);
    args.set(0, distances, AccessModeIds::read);
    args.set(1, categories, AccessModeIds::read);
    args.set(2, partialDistances, AccessModeIds::readwrite);
    args.set(3, partialCategories, AccessModeIds::readwrite);
    args.set(4, nK);
    args.set(5, nPart);
    args.set(6, totalParts);

    KernelRange local_range(1, 1);
    KernelRange global_range(probeBlockSize, nK);

    KernelNDRange range(2);
    range.global(global_range, st); DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st); DAAL_CHECK_STATUS_PTR(st);

    {
        context.run(range, kernel_gather_selection, args, st);
    }
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::initDistances
        (ExecutionContextIface& context,
         const KernelPtr& kernel_init_distances,
         UniversalBuffer& dataSq,
         UniversalBuffer& distances,
         uint32_t dataBlockSize,
         uint32_t probesBlockSize,
         Status* st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initDistances);

    KernelArguments args(3);
    args.set(0, dataSq, AccessModeIds::read);
    args.set(1, distances, AccessModeIds::write);
    args.set(2, dataBlockSize);

    KernelRange global_range(dataBlockSize, probesBlockSize);

    {
        context.run(global_range, kernel_init_distances, args, st);
    }
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::computeDistances
        (ExecutionContextIface& context,
         const Buffer<algorithmFpType>& data,
         const Buffer<algorithmFpType>& probes,
         UniversalBuffer& distances,
         uint32_t dataBlockSize,
         uint32_t probeBlockSize,
         uint32_t nFeatures,
         Status* st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.GEMM);
    auto gemmStatus = BlasGpu<algorithmFpType>::xgemm
            (math::Layout::RowMajor,
             math::Transpose::NoTrans, math::Transpose::Trans,
             probeBlockSize, dataBlockSize, nFeatures,
             algorithmFpType(-2.0),
             probes, nFeatures, 0,
             data, nFeatures, 0,
             algorithmFpType(1.0),
             distances.get<algorithmFpType>(), dataBlockSize, 0);

    if (st != nullptr) { *st = gemmStatus; }
}

template <typename algorithmFpType>
void KNNClassificationPredictKernelUCAPI<algorithmFpType>::computeWinners
        (ExecutionContextIface& context,
         const KernelPtr& kernel_compute_winners,
         UniversalBuffer& categories,
         UniversalBuffer& classes,
         uint32_t probesBlockSize,
         uint32_t nK,
         Status* st)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(compute.initDistances);

    KernelArguments args(3);
    args.set(0, categories, AccessModeIds::read);
    args.set(1, classes, AccessModeIds::write);
    args.set(2, nK);

    KernelRange local_range(1);
    KernelRange global_range(probesBlockSize);

    KernelNDRange range(1);
    range.global(global_range, st); DAAL_CHECK_STATUS_PTR(st);
    range.local(local_range, st); DAAL_CHECK_STATUS_PTR(st);

    {
        context.run(range, kernel_compute_winners, args, st);
    }
}

} // namespace internal
} // namespace prediction
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
