/* file: low_order_moments_distributed_oneapi_impl.i */
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
//  Low order moments algorithm implementation in distributed mode.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_DISTRIBUTED_ONEAPI_IMPL_I__
#define __LOW_ORDER_MOMENTS_DISTRIBUTED_ONEAPI_IMPL_I__

#include "services/internal/buffer.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "services/error_indexes.h"
#include "src/algorithms/low_order_moments/oneapi/cl_kernels/low_order_moments_kernels_distr.h"
#include "src/algorithms/low_order_moments/oneapi/low_order_moments_kernel_distributed_oneapi.h"
#include "src/externals/service_profiler.h"
#include "services/internal/execution_context.h"
#include "services/daal_defines.h"

using namespace daal::services::internal;
using namespace daal::services::internal::sycl;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{
namespace oneapi
{
namespace internal
{
/* task info MinMax parameters definitions */
template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMinMax>::kMergeDistrBlocksName = "mergeDistrBlocksMinMax";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMinMax>::kBldOptFNameSuff = " -DFNAMESUFF=MinMax ";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMinMax>::kBldOptScope = " -D_RMIN_ -D_RMAX_ ";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMinMax>::kCacheKey = "__daal_algorithms_low_order_moments_distributed_kernels_minmax";

/* itask info MeanVariance parameters definitions */
template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMeanVariance>::kMergeDistrBlocksName = "mergeDistrBlocksMeanVariance";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMeanVariance>::kFinalizeName = "finalizeMeanVariance";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMeanVariance>::kBldOptFNameSuff = " -DFNAMESUFF=MeanVariance ";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMeanVariance>::kBldOptScope = " -D_RMEAN_ -D_RVARC_ ";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesMeanVariance>::kCacheKey =
    "__daal_algorithms_low_order_moments_distributed_kernels_mean_variance";

/* All task info estimatesAll parameters definitions */
template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesAll>::kMergeDistrBlocksName = "mergeDistrBlocksAll";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesAll>::kFinalizeName = "finalizeAll";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesAll>::kBldOptFNameSuff = " -DFNAMESUFF=All ";

template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesAll>::kBldOptScope =
    " -D_RMIN_ -D_RMAX_ -D_RSUM_ -D_RSUM2_ -D_RSUM2C_ -D_RMEAN_ -D_RSORM_ -D_RVARC_ -D_RSTDEV_ -D_RVART_ ";
template <>
const char * TaskInfoDistributed<DAAL_FPTYPE, estimatesAll>::kCacheKey = "__daal_algorithms_low_order_moments_distributed_kernels_all";

/*
   Kernel methods implementation
*/

template <typename algorithmFPType, Method method>
services::Status LowOrderMomentsDistributedKernelOneAPI<algorithmFPType, method>::compute(data_management::DataCollection * partialResultsCollection,
                                                                                          PartialResult * partialResult, const Parameter * parameter)
{
    services::Status status;

    auto & context = daal::services::internal::getDefaultContext();

    if (method == defaultDense)
    {
        if (parameter->estimatesToCompute == estimatesMinMax)
        {
            LowOrderMomentsDistributedTaskOneAPI<algorithmFPType, estimatesMinMax> task(context, partialResultsCollection, partialResult, status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
        else if (parameter->estimatesToCompute == estimatesMeanVariance)
        {
            LowOrderMomentsDistributedTaskOneAPI<algorithmFPType, estimatesMeanVariance> task(context, partialResultsCollection, partialResult,
                                                                                              status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
        else
        {
            /* estimatesAll */
            LowOrderMomentsDistributedTaskOneAPI<algorithmFPType, estimatesAll> task(context, partialResultsCollection, partialResult, status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
    }

    return services::Status(ErrorMethodNotImplemented);
}

template <typename algorithmFPType, Method method>
services::Status LowOrderMomentsDistributedKernelOneAPI<algorithmFPType, method>::finalizeCompute(PartialResult * partialResult, Result * result,
                                                                                                  const Parameter * parameter)
{
    services::Status status;

    auto & context = daal::services::internal::getDefaultContext();
    if (method == defaultDense)
    {
        /*nothing is done in case of finalizing results which are already available in partialResult (i.e. min max ...),
          due to they should be already assigned into result from partial results by level up caller
        */
        if (parameter->estimatesToCompute == estimatesMeanVariance)
        {
            LowOrderMomentsDistributedFinalizeTaskOneAPI<algorithmFPType, estimatesMeanVariance> task(context, partialResult, result, status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
        else if (parameter->estimatesToCompute == estimatesAll)
        {
            /* estimatesAll */
            LowOrderMomentsDistributedFinalizeTaskOneAPI<algorithmFPType, estimatesAll> task(context, partialResult, result, status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
    }

    return status;
}

template <typename T, typename Q, typename P>
static inline services::Status overflowCheckByMultiplication(const Q & v1, const P & v2)
{
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(T, v1, v2);
    return services::Status();
}

template <typename algorithmFPType, EstimatesToCompute scope>
static inline services::Status buildProgram(ClKernelFactoryIface & factory, const char * buildOptions = nullptr)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);

    services::Status status;
    auto fptype_name   = getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;

    build_options.add(" -cl-std=CL1.2 -D LOCAL_BUFFER_SIZE=256 ");
    build_options.add(TaskInfoDistributed<algorithmFPType, scope>::kBldOptFNameSuff);
    build_options.add(TaskInfoDistributed<algorithmFPType, scope>::kBldOptScope);

    if (buildOptions)
    {
        build_options.add(buildOptions);
    }

    services::String cachekey(TaskInfoDistributed<algorithmFPType, scope>::kCacheKey);
    cachekey.add(fptype_name);

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), low_order_moments_kernels_distr_cl, build_options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

/*
    Distributed task methods implementations
*/
template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsDistributedTaskOneAPI<algorithmFPType, scope>::LowOrderMomentsDistributedTaskOneAPI(
    ExecutionContextIface & context, data_management::DataCollection * partialResultsCollection, PartialResult * partialResult,
    services::Status & status)
    : partResultsCollection(partialResultsCollection)
{
    auto pRTable = partialResult->get((PartialResultId)TaskInfoDistributed<algorithmFPType, scope>::resPartialIds[0]);
    status |= pRTable ? services::Status() : services::Status(ErrorNullPartialResult);
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

    nFeatures = pRTable->getNumberOfColumns();

    if (partialResultsCollection->size() > _uint32max)
    {
        status |= services::ErrorIncorrectNumberOfElementsInInputCollection;
        return;
    }
    nDistrBlocks = partialResultsCollection->size();
    bNVec        = context.allocate(TypeIds::id<algorithmFPType>(), nDistrBlocks, status);
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

    status |= overflowCheckByMultiplication<size_t>(nFeatures, sizeof(algorithmFPType));
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

    size_t nElemsInStrideST = (((nFeatures * sizeof(algorithmFPType)) + _blockAlignment - 1) & ~(_blockAlignment - 1)) / sizeof(algorithmFPType);
    if (nElemsInStrideST > _uint32max)
    {
        status |= services::ErrorIncorrectNumberOfColumnsInInputNumericTable;
        return;
    }

    nElemsInStride = static_cast<uint32_t>(nElemsInStrideST);

    status |= overflowCheckByMultiplication<size_t>(nDistrBlocks, nElemsInStride);
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

    for (uint32_t i = 0; i < TaskInfoDistributed<algorithmFPType, scope>::nBuffers; i++)
    {
        bAuxHostBuffers[i].reset(nDistrBlocks * nElemsInStride);
        status |= bAuxHostBuffers[i].get() ? services::Status() : services::Status(services::ErrorMemoryAllocationFailed);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        bAuxBuffers[i] = context.allocate(TypeIds::id<algorithmFPType>(), nDistrBlocks * nElemsInStride, status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
    }

    nObservationsTable = partialResult->get((PartialResultId)nObservations);
    status |= nObservationsTable ? nObservationsTable->getBlockOfRows(0, 1, readWrite, nObservationsBD) : services::Status(ErrorNullPartialResult);
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
    pNObservations = nObservationsBD.getBlockPtr();

    for (uint32_t i = 0; i < TaskInfoDistributed<algorithmFPType, scope>::nPartialResults; i++)
    {
        resultTable[i] = partialResult->get((PartialResultId)TaskInfoDistributed<algorithmFPType, scope>::resPartialIds[i]);
        status |= resultTable[i] ? resultTable[i]->getBlockOfRows(0, 1, readWrite, resultBD[i]) : services::Status(ErrorNullPartialResult);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
    }
}

template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsDistributedTaskOneAPI<algorithmFPType, scope>::~LowOrderMomentsDistributedTaskOneAPI()
{
    if (nObservationsTable)
    {
        nObservationsTable->releaseBlockOfRows(nObservationsBD);
    }

    for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nPartialResults; i++)
    {
        if (resultTable[i])
        {
            resultTable[i]->releaseBlockOfRows(resultBD[i]);
        }
    }
}

template <typename algorithmFPType, EstimatesToCompute scope>
services::Status LowOrderMomentsDistributedTaskOneAPI<algorithmFPType, scope>::compute()
{
    DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsDistributedTaskOneAPI.compute);

    DAAL_ASSERT(partResultsCollection);

    services::Status status;

    auto & context = daal::services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();

    status = buildProgram<algorithmFPType, scope>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_MALLOC(pNObservations);
    *pNObservations = (algorithmFPType)0;

    DAAL_ASSERT_UNIVERSAL_BUFFER(bNVec, algorithmFPType, nDistrBlocks);

    {
        auto bNVecHost = bNVec.template get<algorithmFPType>().toHost(ReadWriteMode::writeOnly, status);
        DAAL_CHECK_STATUS_VAR(status);

        for (uint32_t distrBlockId = 0; distrBlockId < nDistrBlocks; distrBlockId++)
        {
            PartialResult * inputPartialResult = static_cast<PartialResult *>((*partResultsCollection)[distrBlockId].get());
            BlockDescriptor<algorithmFPType> blockDesc;

            NumericTablePtr tablePtr = inputPartialResult->get((PartialResultId)nObservations);
            DAAL_CHECK_STATUS_VAR(tablePtr ? tablePtr->getBlockOfRows(0, 1, readOnly, blockDesc) : services::Status(ErrorNullPartialResult));
            algorithmFPType * pBlockObsCount = blockDesc.getBlockPtr();
            DAAL_CHECK_MALLOC(pBlockObsCount);

            bNVecHost.get()[distrBlockId] = *pBlockObsCount;

            *pNObservations += *pBlockObsCount;
            tablePtr->releaseBlockOfRows(blockDesc);
        }
    }

    const size_t rowSize = nFeatures * sizeof(algorithmFPType);

    for (uint32_t i = 0; i < TaskInfoDistributed<algorithmFPType, scope>::nPartialResults; i++)
    {
        // copy partial results from each block into one common buffer
        DAAL_ASSERT(bAuxHostBuffers[i].size() == nDistrBlocks * nElemsInStride);
        for (uint32_t distrBlockId = 0; distrBlockId < nDistrBlocks; distrBlockId++)
        {
            PartialResult * inputPartialResult = static_cast<PartialResult *>((*partResultsCollection)[distrBlockId].get());
            BlockDescriptor<algorithmFPType> blockDesc;
            NumericTablePtr tablePtr = inputPartialResult->get((PartialResultId)TaskInfoDistributed<algorithmFPType, scope>::resPartialIds[i]);
            DAAL_CHECK_STATUS_VAR(tablePtr ? tablePtr->getBlockOfRows(0, 1, readOnly, blockDesc) : services::Status(ErrorNullPartialResult));
            DAAL_ASSERT(blockDesc.getBuffer().size() == nFeatures);
            int result = daal::services::internal::daal_memcpy_s(bAuxHostBuffers[i].get() + distrBlockId * nElemsInStride, rowSize,
                                                                 blockDesc.getBlockPtr(), rowSize);
            DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);
            tablePtr->releaseBlockOfRows(blockDesc);
        }
        DAAL_ASSERT_UNIVERSAL_BUFFER(bAuxBuffers[i], algorithmFPType, nDistrBlocks * nElemsInStride);
        context.copy(bAuxBuffers[i], 0, (void *)bAuxHostBuffers[i].get(), nDistrBlocks * nElemsInStride, 0, nDistrBlocks * nElemsInStride, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    /* merge blocks */
    auto kMergeDistrBlocks = factory.getKernel(TaskInfoDistributed<algorithmFPType, scope>::kMergeDistrBlocksName, status);
    DAAL_CHECK_STATUS_VAR(status);
    {
        KernelRange range(nFeatures);

        KernelArguments args(3 + TaskInfoDistributed<algorithmFPType, scope>::nPartialResults + TaskInfoDistributed<algorithmFPType, scope>::nBuffers
                                 + 1 /*rows in block info*/,
                             status);
        DAAL_CHECK_STATUS_VAR(status);

        uint32_t argsI = 0;
        args.set(argsI++, nFeatures);
        args.set(argsI++, nDistrBlocks);   // num of values to merge
        args.set(argsI++, nElemsInStride); // stride between feature values from different blocks
        for (uint32_t i = 0; i < TaskInfoDistributed<algorithmFPType, scope>::nPartialResults; i++)
        {
            DAAL_ASSERT(resultBD[i].getBuffer().size() == nFeatures);
            args.set(argsI++, resultBD[i].getBuffer(), AccessModeIds::readwrite);
        }

        DAAL_ASSERT_UNIVERSAL_BUFFER(bNVec, algorithmFPType, nDistrBlocks);
        args.set(argsI++, bNVec, AccessModeIds::read);

        for (uint32_t i = 0; i < TaskInfoDistributed<algorithmFPType, scope>::nBuffers; i++)
        {
            DAAL_ASSERT_UNIVERSAL_BUFFER(bAuxBuffers[i], algorithmFPType, nFeatures * nDistrBlocks);
            args.set(argsI++, bAuxBuffers[i], AccessModeIds::read);
        }

        {
            DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsDistributedTaskOneAPI.MergeDistrBlocks);
            context.run(range, kMergeDistrBlocks, args, status);
        }
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}
/*
    finalize task methods implementations
*/
template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsDistributedFinalizeTaskOneAPI<algorithmFPType, scope>::LowOrderMomentsDistributedFinalizeTaskOneAPI(ExecutionContextIface & context,
                                                                                                                   PartialResult * partialResult,
                                                                                                                   Result * result,
                                                                                                                   services::Status & status)
{
    uint32_t resIdx = 0;
    for (uint32_t i = 0; i < TaskInfoDistributed<algorithmFPType, scope>::nPartialResults; i++)
    {
        resultTable[resIdx] = partialResult->get((PartialResultId)TaskInfoDistributed<algorithmFPType, scope>::resPartialIds[i]);
        status |=
            resultTable[resIdx] ? resultTable[resIdx]->getBlockOfRows(0, 1, readWrite, resultBD[resIdx]) : services::Status(ErrorNullPartialResult);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
        resIdx++;
    }

    for (uint32_t i = 0; i < TaskInfoDistributed<algorithmFPType, scope>::nFinalizeResults; i++)
    {
        resultTable[resIdx] = result->get((ResultId)TaskInfoDistributed<algorithmFPType, scope>::resFinalizeIds[i]);
        status |=
            resultTable[resIdx] ? resultTable[resIdx]->getBlockOfRows(0, 1, readWrite, resultBD[resIdx]) : services::Status(ErrorNullPartialResult);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
        resIdx++;
    }

    nFeatures = resultTable[0]->getNumberOfColumns();

    nObservationsTable = partialResult->get((PartialResultId)nObservations);
    status |= nObservationsTable ? nObservationsTable->getBlockOfRows(0, 1, readWrite, nObservationsBD) : services::Status(ErrorNullPartialResult);
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
    pNObservations = nObservationsBD.getBlockPtr();
}

template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsDistributedFinalizeTaskOneAPI<algorithmFPType, scope>::~LowOrderMomentsDistributedFinalizeTaskOneAPI()
{
    if (nObservationsTable)
    {
        nObservationsTable->releaseBlockOfRows(nObservationsBD);
    }

    for (uint32_t i = 0; i < nTotalResults; i++)
    {
        if (resultTable[i])
        {
            resultTable[i]->releaseBlockOfRows(resultBD[i]);
        }
    }
}

template <typename algorithmFPType, EstimatesToCompute scope>
services::Status LowOrderMomentsDistributedFinalizeTaskOneAPI<algorithmFPType, scope>::compute()
{
    DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsDistributedTaskOneAPI.finalize);

    services::Status status;

    auto & context = daal::services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();

    status = buildProgram<algorithmFPType, scope>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kFinalize = factory.getKernel(TaskInfoDistributed<algorithmFPType, scope>::kFinalizeName, status);
    DAAL_CHECK_STATUS_VAR(status);
    {
        KernelRange range(nFeatures);

        KernelArguments args(1 + nTotalResults, status);
        DAAL_CHECK_STATUS_VAR(status);

        uint32_t argsI = 0;
        DAAL_CHECK_MALLOC(pNObservations);
        args.set(argsI++, *pNObservations);

        for (uint32_t i = 0; i < nTotalResults; i++)
        {
            DAAL_ASSERT(resultBD[i].getBuffer().size() == nFeatures);
            args.set(argsI++, resultBD[i].getBuffer(),
                     (i < TaskInfoDistributed<algorithmFPType, scope>::nPartialResults ? AccessModeIds::read : AccessModeIds::write));
        }

        context.run(range, kFinalize, args, status);
        DAAL_CHECK_STATUS_VAR(status);
    }

    return status;
}

} // namespace internal
} // namespace oneapi
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
