/* file: low_order_moments_online_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Low order moments algorithm implementation in online mode.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_ONLINE_ONEAPI_IMPL_I__
#define __LOW_ORDER_MOMENTS_ONLINE_ONEAPI_IMPL_I__

#include "services/internal/buffer.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "services/error_indexes.h"
#include "src/algorithms/low_order_moments/oneapi/cl_kernels/low_order_moments_kernels_all.h"
#include "src/algorithms/low_order_moments/oneapi/low_order_moments_kernel_online_oneapi.h"
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
#define CHECK_AND_RET_IF_FAIL(st, expr) \
    (st) |= (expr);                     \
    if (!st)                            \
    {                                   \
        return;                         \
    }

/* task info MinMax parameters definitions */
template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMinMax>::kSinglePassName = "singlePassMinMax";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMinMax>::kProcessBlocksName = "processBlocksMinMax";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMinMax>::kMergeBlocksName = "mergeBlocksMinMax";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMinMax>::kBldOptFNameSuff = " -DFNAMESUFF=MinMax ";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMinMax>::kBldOptScope = " -D_ONLINE_ -D_RMIN_ -D_RMAX_ ";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMinMax>::kCacheKey = "__daal_algorithms_low_order_moments_online_kernels_minmax";

/* itask info MeanVariance parameters definitions */
template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMeanVariance>::kSinglePassName = "singlePassMeanVariance";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMeanVariance>::kProcessBlocksName = "processBlocksMeanVariance";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMeanVariance>::kMergeBlocksName = "mergeBlocksMeanVariance";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMeanVariance>::kFinalizeName = "finalizeMeanVariance";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMeanVariance>::kBldOptFNameSuff = " -DFNAMESUFF=MeanVariance ";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMeanVariance>::kBldOptScope = " -D_ONLINE_ -D_RMEAN_ -D_RVARC_ ";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesMeanVariance>::kCacheKey = "__daal_algorithms_low_order_moments_online_kernels_mean_variance";

/* All task info estimatesAll parameters definitions */
template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesAll>::kSinglePassName = "singlePassAll";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesAll>::kProcessBlocksName = "processBlocksAll";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesAll>::kMergeBlocksName = "mergeBlocksAll";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesAll>::kFinalizeName = "finalizeAll";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesAll>::kBldOptFNameSuff = " -DFNAMESUFF=All ";

template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesAll>::kBldOptScope =
    " -D_ONLINE_ -D_RMIN_ -D_RMAX_ -D_RSUM_ -D_RSUM2_ -D_RSUM2C_ -D_RMEAN_ -D_RSORM_ -D_RVARC_ -D_RSTDEV_ -D_RVART_ ";
template <>
const char * TaskInfoOnline<DAAL_FPTYPE, estimatesAll>::kCacheKey = "__daal_algorithms_low_order_moments_online_kernels_all";

/*
   Kernel methods implementation
*/
template <typename algorithmFPType, Method method>
services::Status LowOrderMomentsOnlineKernelOneAPI<algorithmFPType, method>::compute(NumericTable * dataTable, PartialResult * partialResult,
                                                                                     const Parameter * parameter, bool isOnline)
{
    services::Status status;

    auto & context = daal::services::internal::getDefaultContext();

    if (method == defaultDense)
    {
        if (parameter->estimatesToCompute == estimatesMinMax)
        {
            LowOrderMomentsOnlineTaskOneAPI<algorithmFPType, estimatesMinMax> task(context, dataTable, partialResult, status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
        else if (parameter->estimatesToCompute == estimatesMeanVariance)
        {
            LowOrderMomentsOnlineTaskOneAPI<algorithmFPType, estimatesMeanVariance> task(context, dataTable, partialResult, status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
        else
        {
            /* estimatesAll */
            LowOrderMomentsOnlineTaskOneAPI<algorithmFPType, estimatesAll> task(context, dataTable, partialResult, status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
    }

    return services::Status(ErrorMethodNotImplemented);
}

template <typename algorithmFPType, Method method>
services::Status LowOrderMomentsOnlineKernelOneAPI<algorithmFPType, method>::finalizeCompute(PartialResult * partialResult, Result * result,
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
            LowOrderMomentsOnlineFinalizeTaskOneAPI<algorithmFPType, estimatesMeanVariance> task(context, partialResult, result, status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
        else if (parameter->estimatesToCompute == estimatesAll)
        {
            /* estimatesAll */
            LowOrderMomentsOnlineFinalizeTaskOneAPI<algorithmFPType, estimatesAll> task(context, partialResult, result, status);
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
    build_options.add(TaskInfoOnline<algorithmFPType, scope>::kBldOptFNameSuff);
    build_options.add(TaskInfoOnline<algorithmFPType, scope>::kBldOptScope);

    if (buildOptions)
    {
        build_options.add(buildOptions);
    }

    services::String cachekey(TaskInfoOnline<algorithmFPType, scope>::kCacheKey);
    cachekey.add(fptype_name);

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), low_order_moments_kernels_all_cl, build_options.c_str(), status);
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

/*
    Online task methods implementations
*/
template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsOnlineTaskOneAPI<algorithmFPType, scope>::LowOrderMomentsOnlineTaskOneAPI(ExecutionContextIface & context, NumericTable * dataTable,
                                                                                         PartialResult * partialResult, services::Status & status)
    : dataTable(dataTable)
{
    if (dataTable->getNumberOfRows() > _uint32max)
    {
        status |= services::ErrorIncorrectNumberOfRowsInInputNumericTable;
        return;
    }
    if (dataTable->getNumberOfColumns() > _uint32max)
    {
        status |= services::ErrorIncorrectNumberOfColumnsInInputNumericTable;
        return;
    }

    nVectors  = static_cast<uint32_t>(dataTable->getNumberOfRows());
    nFeatures = static_cast<uint32_t>(dataTable->getNumberOfColumns());

    nColsBlocks = (nFeatures + maxWorkItemsPerGroup - 1) / maxWorkItemsPerGroup;

    nRowsBlocks = 128;
    if (nVectors < 5000)
        nRowsBlocks = 1;
    else if (nVectors < 10000)
        nRowsBlocks = 8;
    else if (nVectors < 20000)
        nRowsBlocks = 16;
    else if (nVectors < 50000)
        nRowsBlocks = 32;
    else if (nVectors < 100000)
        nRowsBlocks = 64;

    workItemsPerGroup = (maxWorkItemsPerGroup < nFeatures) ? maxWorkItemsPerGroup : nFeatures;

    status |= dataTable->getBlockOfRows(0, nVectors, readOnly, dataBD);
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

    nObservationsTable = partialResult->get((PartialResultId)nObservations);
    status |= nObservationsTable ? nObservationsTable->getBlockOfRows(0, 1, readWrite, nObservationsBD) : services::Status(ErrorNullPartialResult);
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
    pNObservations = nObservationsBD.getBlockPtr();

    for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nPartialResults; i++)
    {
        resultTable[i] = partialResult->get((PartialResultId)TaskInfoOnline<algorithmFPType, scope>::resPartialIds[i]);
        status |= resultTable[i]->getBlockOfRows(0, 1, readWrite, resultBD[i]);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
    }

    status |= overflowCheckByMultiplication<size_t>(nRowsBlocks, nFeatures);
    DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

    if (TaskInfoOnline<algorithmFPType, scope>::isRowsInBlockInfoRequired)
    {
        if (nRowsBlocks > 1)
        {
            bNVec = context.allocate(TypeIds::uint32, nFeatures * nRowsBlocks, status);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
        }
    }

    if (nRowsBlocks > 1)
    {
        for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nBuffers; i++)
        {
            bAuxBuffers[i] = context.allocate(TypeIds::id<algorithmFPType>(), nFeatures * nRowsBlocks, status);
            DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
        }
    }
}

template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsOnlineTaskOneAPI<algorithmFPType, scope>::~LowOrderMomentsOnlineTaskOneAPI()
{
    if (dataTable)
    {
        dataTable->releaseBlockOfRows(dataBD);
    }

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
services::Status LowOrderMomentsOnlineTaskOneAPI<algorithmFPType, scope>::compute()
{
    DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsOnlineTaskOneAPI.compute);

    services::Status status;

    auto & context = daal::services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();

    status = buildProgram<algorithmFPType, scope>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_MALLOC(pNObservations);

    if (nRowsBlocks > 1)
    {
        /* process rows by blocks first */
        auto kProcessBlocks = factory.getKernel(TaskInfoOnline<algorithmFPType, scope>::kProcessBlocksName, status);
        DAAL_CHECK_STATUS_VAR(status);
        {
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsBlocks, nColsBlocks);
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRowsBlocks * nColsBlocks, workItemsPerGroup);
            KernelRange localRange(workItemsPerGroup);
            KernelRange globalRange(nRowsBlocks * nColsBlocks * workItemsPerGroup);

            KernelNDRange range(1);
            range.global(globalRange, status);
            DAAL_CHECK_STATUS_VAR(status);
            range.local(localRange, status);
            DAAL_CHECK_STATUS_VAR(status);

            KernelArguments args(
                3 + TaskInfoOnline<algorithmFPType, scope>::nBuffers + (TaskInfoOnline<algorithmFPType, scope>::isRowsInBlockInfoRequired ? 1 : 0),
                status);
            DAAL_CHECK_STATUS_VAR(status);

            uint32_t argsI = 0;
            DAAL_ASSERT(dataBD.getBuffer().size() == nVectors * nFeatures);
            args.set(argsI++, dataBD.getBuffer(), AccessModeIds::read);
            args.set(argsI++, nFeatures);
            args.set(argsI++, nVectors);

            if (TaskInfoOnline<algorithmFPType, scope>::isRowsInBlockInfoRequired)
            {
                DAAL_ASSERT_UNIVERSAL_BUFFER(bNVec, uint32_t, nFeatures * nRowsBlocks);
                args.set(argsI++, bNVec, AccessModeIds::write);
            }

            for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nBuffers; i++)
            {
                DAAL_ASSERT_UNIVERSAL_BUFFER(bAuxBuffers[i], algorithmFPType, nFeatures * nRowsBlocks);
                args.set(argsI++, bAuxBuffers[i], AccessModeIds::write);
            }

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsOnlineTaskOneAPI.ProcessBlocks);
                context.run(range, kProcessBlocks, args, status);
            }
            DAAL_CHECK_STATUS_VAR(status);
        }

        /* merge blocks */
        auto kMergeBlocks = factory.getKernel(TaskInfoOnline<algorithmFPType, scope>::kMergeBlocksName, status);
        DAAL_CHECK_STATUS_VAR(status);
        {
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, maxWorkItemsPerGroupToMerge);
            KernelRange localRange(maxWorkItemsPerGroupToMerge);
            KernelRange globalRange(maxWorkItemsPerGroupToMerge * nFeatures);

            KernelNDRange range(1);
            range.global(globalRange, status);
            DAAL_CHECK_STATUS_VAR(status);
            range.local(localRange, status);
            DAAL_CHECK_STATUS_VAR(status);

            KernelArguments args(2 + TaskInfoOnline<algorithmFPType, scope>::nPartialResults + TaskInfoOnline<algorithmFPType, scope>::nBuffers
                                     + (TaskInfoOnline<algorithmFPType, scope>::isRowsInBlockInfoRequired ? 1 : 0),
                                 status);
            DAAL_CHECK_STATUS_VAR(status);

            uint32_t argsI = 0;
            args.set(argsI++, nRowsBlocks); // num of values to merge
            args.set(argsI++, *pNObservations);
            for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nPartialResults; i++)
            {
                DAAL_ASSERT(resultBD[i].getBuffer().size() == nFeatures);
                args.set(argsI++, resultBD[i].getBuffer(), AccessModeIds::readwrite);
            }

            if (TaskInfoOnline<algorithmFPType, scope>::isRowsInBlockInfoRequired)
            {
                DAAL_ASSERT_UNIVERSAL_BUFFER(bNVec, uint32_t, nFeatures * nRowsBlocks);
                args.set(argsI++, bNVec, AccessModeIds::write);
            }

            for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nBuffers; i++)
            {
                DAAL_ASSERT_UNIVERSAL_BUFFER(bAuxBuffers[i], algorithmFPType, nFeatures * nRowsBlocks);
                args.set(argsI++, bAuxBuffers[i], AccessModeIds::write);
            }

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsOnlineTaskOneAPI.MergeBlocks);
                context.run(range, kMergeBlocks, args, status);
            }
            DAAL_CHECK_STATUS_VAR(status);
        }
    }
    else
    {
        auto kSinglePass = factory.getKernel(TaskInfoOnline<algorithmFPType, scope>::kSinglePassName, status);
        DAAL_CHECK_STATUS_VAR(status);
        {
            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nColsBlocks, workItemsPerGroup);
            KernelRange localRange(workItemsPerGroup);
            KernelRange globalRange(nColsBlocks * workItemsPerGroup);

            KernelNDRange range(1);
            range.global(globalRange, status);
            DAAL_CHECK_STATUS_VAR(status);
            range.local(localRange, status);
            DAAL_CHECK_STATUS_VAR(status);

            KernelArguments args(4 + TaskInfoOnline<algorithmFPType, scope>::nPartialResults, status);
            DAAL_CHECK_STATUS_VAR(status);

            uint32_t argsI = 0;
            DAAL_ASSERT(dataBD.getBuffer().size() == nVectors * nFeatures);
            args.set(argsI++, dataBD.getBuffer(), AccessModeIds::read);
            args.set(argsI++, nFeatures);
            args.set(argsI++, nVectors);
            args.set(argsI++, *pNObservations);
            for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nPartialResults; i++)
            {
                DAAL_ASSERT(resultBD[i].getBuffer().size() == nFeatures);
                args.set(argsI++, resultBD[i].getBuffer(), AccessModeIds::readwrite);
            }

            context.run(range, kSinglePass, args, status);
            DAAL_CHECK_STATUS_VAR(status);
        }
    }

    *pNObservations += static_cast<algorithmFPType>(nVectors);

    return status;
}
/*
    finalize task methods implementations
*/
template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsOnlineFinalizeTaskOneAPI<algorithmFPType, scope>::LowOrderMomentsOnlineFinalizeTaskOneAPI(ExecutionContextIface & context,
                                                                                                         PartialResult * partialResult,
                                                                                                         Result * result, services::Status & status)
{
    uint32_t resIdx = 0;
    for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nPartialResults; i++)
    {
        resultTable[resIdx] = partialResult->get((PartialResultId)TaskInfoOnline<algorithmFPType, scope>::resPartialIds[i]);
        status |= resultTable[resIdx]->getBlockOfRows(0, 1, readOnly, resultBD[resIdx]);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);
        resIdx++;
    }
    for (uint32_t i = 0; i < TaskInfoOnline<algorithmFPType, scope>::nFinalizeResults; i++)
    {
        resultTable[resIdx] = result->get((ResultId)TaskInfoOnline<algorithmFPType, scope>::resFinalizeIds[i]);
        status |= resultTable[resIdx]->getBlockOfRows(0, 1, readWrite, resultBD[resIdx]);
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
LowOrderMomentsOnlineFinalizeTaskOneAPI<algorithmFPType, scope>::~LowOrderMomentsOnlineFinalizeTaskOneAPI()
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
services::Status LowOrderMomentsOnlineFinalizeTaskOneAPI<algorithmFPType, scope>::compute()
{
    DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsOnlineTaskOneAPI.finalize);

    services::Status status;

    auto & context = daal::services::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();

    status = buildProgram<algorithmFPType, scope>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    auto kFinalize = factory.getKernel(TaskInfoOnline<algorithmFPType, scope>::kFinalizeName, status);
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
                     (i < TaskInfoOnline<algorithmFPType, scope>::nPartialResults ? AccessModeIds::read : AccessModeIds::write));
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
