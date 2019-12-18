/* file: low_order_moments_batch_oneapi_impl.i */
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

/*
//++
//  Low order moments algorithm implementation in batch mode.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_BATCH_ONEAPI_IMPL_I__
#define __LOW_ORDER_MOMENTS_BATCH_ONEAPI_IMPL_I__

#include "services/buffer.h"
#include "numeric_table.h"
#include "env_detect.h"
#include "error_indexes.h"
#include "cl_kernels/low_order_moments_kernels_all.h"
#include "low_order_moments_kernel_batch_oneapi.h"
#include "service_ittnotify.h"
#include "oneapi/internal/utils.h"

using namespace daal::services::internal;
using namespace daal::oneapi::internal;

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
#define RET_IF_FAIL(st) \
    if (!st)            \
    {                   \
        return;         \
    }
#define CHECK_AND_RET_IF_FAIL(st, expr) \
    (st) |= (expr);                     \
    if (!st)                            \
    {                                   \
        return;                         \
    }

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMinMax>::kSinglePassName = "singlePassMinMax";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMinMax>::kProcessBlocksName = "processBlocksMinMax";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMinMax>::kMergeBlocksName = "mergeBlocksMinMax";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMinMax>::kBldOptFNameSuff = " -DFNAMESUFF=MinMax ";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMinMax>::kBldOptScope = " -D_RMIN_ -D_RMAX_ ";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMinMax>::kCacheKey = "__daal_algorithms_low_order_moments_batch_kernels_minmax";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMeanVariance>::kSinglePassName = "singlePassMeanVariance";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMeanVariance>::kProcessBlocksName = "processBlocksMeanVariance";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMeanVariance>::kMergeBlocksName = "mergeBlocksMeanVariance";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMeanVariance>::kBldOptFNameSuff = " -DFNAMESUFF=MeanVariance ";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMeanVariance>::kBldOptScope = " -D_RMEAN_ -D_RVARC_ ";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesMeanVariance>::kCacheKey = "__daal_algorithms_low_order_moments_batch_kernels_mean_variance";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesAll>::kSinglePassName = "singlePassAll";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesAll>::kProcessBlocksName = "processBlocksAll";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesAll>::kMergeBlocksName = "mergeBlocksAll";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesAll>::kBldOptFNameSuff = " -DFNAMESUFF=All ";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesAll>::kBldOptScope =
    " -D_RMIN_ -D_RMAX_ -D_RSUM_ -D_RSUM2_ -D_RSUM2C_ -D_RMEAN_ -D_RSORM_ -D_RVARC_ -D_RSTDEV_ -D_RVART_ ";

template <>
const char * TaskInfoBatch<DAAL_FPTYPE, estimatesAll>::kCacheKey = "__daal_algorithms_low_order_moments_batch_kernels_all";

/*
   Kernel methods implementation
*/
template <typename algorithmFPType, Method method>
services::Status LowOrderMomentsBatchKernelOneAPI<algorithmFPType, method>::compute(NumericTable * dataTable, Result * result,
                                                                                    const Parameter * parameter)
{
    services::Status status;

    auto & context = daal::oneapi::internal::getDefaultContext();

    if (method == defaultDense)
    {
        if (parameter->estimatesToCompute == estimatesMinMax)
        {
            LowOrderMomentsBatchTaskOneAPI<algorithmFPType, estimatesMinMax> task(context, dataTable, result, &status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
        else if (parameter->estimatesToCompute == estimatesMeanVariance)
        {
            LowOrderMomentsBatchTaskOneAPI<algorithmFPType, estimatesMeanVariance> task(context, dataTable, result, &status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
        else
        {
            /* estimatesAll */
            LowOrderMomentsBatchTaskOneAPI<algorithmFPType, estimatesAll> task(context, dataTable, result, &status);
            DAAL_CHECK_STATUS_VAR(status);
            return task.compute();
        }
    }

    return status;
}

template <typename algorithmFPType, EstimatesToCompute scope>
static inline services::Status buildProgram(ClKernelFactoryIface & factory, const char * buildOptions = nullptr)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(buildProgram);

    services::Status status;
    auto fptype_name   = getKeyFPType<algorithmFPType>();
    auto build_options = fptype_name;

    build_options.add(" -cl-std=CL1.2 -D LOCAL_BUFFER_SIZE=256 ");
    build_options.add(TaskInfoBatch<algorithmFPType, scope>::kBldOptFNameSuff);
    build_options.add(TaskInfoBatch<algorithmFPType, scope>::kBldOptScope);

    if (buildOptions)
    {
        build_options.add(buildOptions);
    }

    services::String cachekey(TaskInfoBatch<algorithmFPType, scope>::kCacheKey);
    cachekey.add(fptype_name);

    factory.build(ExecutionTargetIds::device, cachekey.c_str(), low_order_moments_kernels_all_cl, build_options.c_str(), &status);

    return status;
}

/*
    Batch task methods implementations
*/
template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsBatchTaskOneAPI<algorithmFPType, scope>::LowOrderMomentsBatchTaskOneAPI(ExecutionContextIface & context, NumericTable * dataTable,
                                                                                       Result * result, services::Status * status)
    : dataTable(dataTable)
{
    nVectors  = dataTable->getNumberOfRows();
    nFeatures = dataTable->getNumberOfColumns();

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

    CHECK_AND_RET_IF_FAIL(*status, dataTable->getBlockOfRows(0, nVectors, readOnly, dataBD));

    for (unsigned int i = 0; i < TaskInfoBatch<algorithmFPType, scope>::nResults; i++)
    {
        resultTable[i] = result->get((ResultId)TaskInfoBatch<algorithmFPType, scope>::resIds[i]);
        CHECK_AND_RET_IF_FAIL(*status, resultTable[i]->getBlockOfRows(0, 1, writeOnly, resultBD[i]));
    }

    if (TaskInfoBatch<algorithmFPType, scope>::isRowsInBlockInfoRequired)
    {
        if (nRowsBlocks > 1)
        {
            bNVec = context.allocate(TypeIds::uint32, nFeatures * nRowsBlocks, status);
            RET_IF_FAIL(*status);
        }
    }

    if (nRowsBlocks > 1)
    {
        for (unsigned int i = 0; i < TaskInfoBatch<algorithmFPType, scope>::nBuffers; i++)
        {
            bAuxBuffers[i] = context.allocate(TypeIds::id<algorithmFPType>(), nFeatures * nRowsBlocks, status);
            RET_IF_FAIL(*status);
        }
    }
}

template <typename algorithmFPType, EstimatesToCompute scope>
LowOrderMomentsBatchTaskOneAPI<algorithmFPType, scope>::~LowOrderMomentsBatchTaskOneAPI()
{
    if (dataTable)
    {
        dataTable->releaseBlockOfRows(dataBD);
    }

    for (unsigned int i = 0; i < TaskInfoBatch<algorithmFPType, scope>::nResults; i++)
    {
        if (resultTable[i])
        {
            resultTable[i]->releaseBlockOfRows(resultBD[i]);
        }
    }
}

template <typename algorithmFPType, EstimatesToCompute scope>
services::Status LowOrderMomentsBatchTaskOneAPI<algorithmFPType, scope>::compute()
{
    DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsBatchTaskOneAPI.compute);

    services::Status status;

    auto & context = daal::oneapi::internal::getDefaultContext();
    auto & factory = context.getClKernelFactory();

    status = buildProgram<algorithmFPType, scope>(factory);
    DAAL_CHECK_STATUS_VAR(status);

    if (nRowsBlocks > 1)
    {
        /* process rows by blocks first */
        auto kProcessBlocks = factory.getKernel(TaskInfoBatch<algorithmFPType, scope>::kProcessBlocksName);
        {
            KernelRange localRange(workItemsPerGroup);
            KernelRange globalRange(nRowsBlocks * nColsBlocks * workItemsPerGroup);

            KernelNDRange range(1);
            range.global(globalRange, &status);
            DAAL_CHECK_STATUS_VAR(status);
            range.local(localRange, &status);
            DAAL_CHECK_STATUS_VAR(status);

            KernelArguments args(3 + TaskInfoBatch<algorithmFPType, scope>::nBuffers
                                 + (TaskInfoBatch<algorithmFPType, scope>::isRowsInBlockInfoRequired ? 1 : 0));

            unsigned int argsI = 0;
            args.set(argsI++, dataBD.getBuffer(), AccessModeIds::read);
            args.set(argsI++, nFeatures);
            args.set(argsI++, nVectors);

            if (TaskInfoBatch<algorithmFPType, scope>::isRowsInBlockInfoRequired)
            {
                args.set(argsI++, bNVec, AccessModeIds::write);
            }

            for (unsigned int i = 0; i < TaskInfoBatch<algorithmFPType, scope>::nBuffers; i++)
            {
                args.set(argsI++, bAuxBuffers[i], AccessModeIds::write);
            }

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsBatchTaskOneAPI.ProcessBlocks);
                context.run(range, kProcessBlocks, args, &status);
            }
            DAAL_CHECK_STATUS_VAR(status);
        }

        /* merge blocks */
        auto kMergeBlocks = factory.getKernel(TaskInfoBatch<algorithmFPType, scope>::kMergeBlocksName);
        {
            KernelRange localRange(maxWorkItemsPerGroupToMerge);
            KernelRange globalRange(maxWorkItemsPerGroupToMerge * nFeatures);

            KernelNDRange range(1);
            range.global(globalRange, &status);
            DAAL_CHECK_STATUS_VAR(status);
            range.local(localRange, &status);
            DAAL_CHECK_STATUS_VAR(status);

            KernelArguments args(1 + TaskInfoBatch<algorithmFPType, scope>::nResults + TaskInfoBatch<algorithmFPType, scope>::nBuffers
                                 + (TaskInfoBatch<algorithmFPType, scope>::isRowsInBlockInfoRequired ? 1 : 0));

            unsigned int argsI = 0;
            args.set(argsI++, nRowsBlocks); // num of values to merge
            for (unsigned int i = 0; i < TaskInfoBatch<algorithmFPType, scope>::nResults; i++)
            {
                args.set(argsI++, resultBD[i].getBuffer(), AccessModeIds::readwrite);
            }

            if (TaskInfoBatch<algorithmFPType, scope>::isRowsInBlockInfoRequired)
            {
                args.set(argsI++, bNVec, AccessModeIds::write);
            }

            for (unsigned int i = 0; i < TaskInfoBatch<algorithmFPType, scope>::nBuffers; i++)
            {
                args.set(argsI++, bAuxBuffers[i], AccessModeIds::write);
            }

            {
                DAAL_ITTNOTIFY_SCOPED_TASK(LowOrderMomentsBatchTaskOneAPI.MergeBlocks);
                context.run(range, kMergeBlocks, args, &status);
            }
            DAAL_CHECK_STATUS_VAR(status);
        }
    }
    else
    {
        auto kSinglePass = factory.getKernel(TaskInfoBatch<algorithmFPType, scope>::kSinglePassName);
        {
            KernelRange localRange(workItemsPerGroup);
            KernelRange globalRange(nColsBlocks * workItemsPerGroup);

            KernelNDRange range(1);
            range.global(globalRange, &status);
            DAAL_CHECK_STATUS_VAR(status);
            range.local(localRange, &status);
            DAAL_CHECK_STATUS_VAR(status);

            KernelArguments args(3 + TaskInfoBatch<algorithmFPType, scope>::nResults);

            unsigned int argsI = 0;
            args.set(argsI++, dataBD.getBuffer(), AccessModeIds::read);
            args.set(argsI++, nFeatures);
            args.set(argsI++, nVectors);
            for (unsigned int i = 0; i < TaskInfoBatch<algorithmFPType, scope>::nResults; i++)
            {
                args.set(argsI++, resultBD[i].getBuffer(), AccessModeIds::readwrite);
            }

            context.run(range, kSinglePass, args, &status);
            DAAL_CHECK_STATUS_VAR(status);
        }
    }

    return status;
}

} // namespace internal
} // namespace oneapi
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal

#endif
