/* file: linear_model_train_normeq_finalize_oneapi_impl.i */
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
//  Implementation of common base classes for normal equations model training.
//--
*/

#include "linear_model_train_normeq_kernel_oneapi.h"
#include "oneapi/internal/math/types.h"
#include "oneapi/lapack_gpu.h"
#include "service_lapack.h"
#include "service_ittnotify.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace normal_equations
{
namespace training
{
namespace internal
{
using namespace daal::oneapi::internal;

template <typename algorithmFPType>
services::Status FinalizeKernelOneAPI<algorithmFPType>::compute(NumericTable & xtxTable, NumericTable & xtyTable, NumericTable & xtxFinalTable,
                                                                NumericTable & xtyFinalTable, NumericTable & betaTable, bool interceptFlag,
                                                                const KernelHelperOneAPIIface<algorithmFPType> & helper)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(computeFinalize);
    services::Status status;

    const size_t nBetasIntercept = xtxTable.getNumberOfRows();
    const size_t nBetas          = interceptFlag ? nBetasIntercept : (nBetasIntercept + 1);
    const size_t nResponses      = xtyTable.getNumberOfRows();

    {
        if (&xtxTable != &xtxFinalTable)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeFinalize.copyToxtxFinalTable);
            DAAL_CHECK_STATUS(status, copyDataToFinalTable(xtxTable, xtxFinalTable));
        }

        if (&xtyTable != &xtyFinalTable)
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeFinalize.copyToxtyFinalTable);
            DAAL_CHECK_STATUS(status, copyDataToFinalTable(xtyTable, xtyFinalTable));
        }
    }

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

    {
        BlockDescriptor<algorithmFPType> xtxBlock;
        BlockDescriptor<algorithmFPType> xtyBlock;

        DAAL_CHECK_STATUS(status, xtxTable.getBlockOfRows(0, nBetasIntercept, ReadWriteMode::readOnly, xtxBlock));
        DAAL_CHECK_STATUS(status, xtyTable.getBlockOfRows(0, nResponses, ReadWriteMode::readOnly, xtyBlock));

        const services::Buffer<algorithmFPType> xtxBuf = xtxBlock.getBuffer();
        const services::Buffer<algorithmFPType> xtyBuf = xtyBlock.getBuffer();

        DAAL_CHECK_STATUS(status, xtxTable.releaseBlockOfRows(xtxBlock));
        DAAL_CHECK_STATUS(status, xtyTable.releaseBlockOfRows(xtyBlock));

        const TypeIds::Id idType = TypeIds::id<algorithmFPType>();

        UniversalBuffer xtxCopyAlloc = context.allocate(idType, nBetasIntercept * nBetasIntercept, &status);
        DAAL_CHECK_STATUS_VAR(status);

        services::Buffer<algorithmFPType> xtxBufCopy = xtxCopyAlloc.get<algorithmFPType>();
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeFinalize.xtxCopy);
            context.copy(xtxBufCopy, 0, xtxBuf, 0, nBetasIntercept * nBetasIntercept, &status);
        }
        DAAL_CHECK_STATUS_VAR(status);

        UniversalBuffer xtyCopyAlloc = context.allocate(idType, nResponses * nBetasIntercept, &status);
        DAAL_CHECK_STATUS_VAR(status);

        services::Buffer<algorithmFPType> betaBuf = xtyCopyAlloc.get<algorithmFPType>();
        {
            DAAL_ITTNOTIFY_SCOPED_TASK(computeFinalize.betaBufCopy);
            context.copy(betaBuf, 0, xtyBuf, 0, nResponses * nBetasIntercept, &status);
        }
        DAAL_CHECK_STATUS_VAR(status);

        DAAL_CHECK_STATUS(status, helper.computeBetasImpl(nBetasIntercept, xtxBufCopy, nResponses, betaBuf, interceptFlag));

        BlockDescriptor<algorithmFPType> betaBlock;
        DAAL_CHECK_STATUS(status, betaTable.getBlockOfRows(0, nResponses, ReadWriteMode::readWrite, betaBlock));
        services::Buffer<algorithmFPType> betaResBuf = betaBlock.getBuffer();

        DAAL_ITTNOTIFY_SCOPED_TASK(computeFinalize.copyBetaToResult);
        DAAL_CHECK_STATUS(status, helper.copyBetaToResult(betaBuf, betaResBuf, nBetas, nResponses, interceptFlag));
        DAAL_CHECK_STATUS(status, betaTable.releaseBlockOfRows(betaBlock));
    }

    return status;
}

template <typename algorithmFPType>
services::Status FinalizeKernelOneAPI<algorithmFPType>::copyDataToFinalTable(NumericTable & srcTable, NumericTable & dstTable)
{
    services::Status status;
    BlockDescriptor<algorithmFPType> srcBlock;
    BlockDescriptor<algorithmFPType> dstBlock;

    const size_t nRows = srcTable.getNumberOfRows();
    const size_t nCols = srcTable.getNumberOfColumns();

    DAAL_CHECK_STATUS(status, srcTable.getBlockOfRows(0, nRows, ReadWriteMode::readOnly, srcBlock));
    DAAL_CHECK_STATUS(status, dstTable.getBlockOfRows(0, nRows, ReadWriteMode::readWrite, dstBlock));

    const services::Buffer<algorithmFPType> srcBuf = srcBlock.getBuffer();
    services::Buffer<algorithmFPType> dstBuf       = dstBlock.getBuffer();

    auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
    context.copy(dstBuf, 0, srcBuf, 0, nCols * nRows, &status);
    DAAL_CHECK_STATUS_VAR(status);

    DAAL_CHECK_STATUS(status, srcTable.releaseBlockOfRows(srcBlock));
    DAAL_CHECK_STATUS(status, dstTable.releaseBlockOfRows(dstBlock));

    return status;
}

template <typename algorithmFPType>
services::Status FinalizeKernelOneAPI<algorithmFPType>::solveSystem(const size_t p, services::Buffer<algorithmFPType> & a, const size_t ny,
                                                                    services::Buffer<algorithmFPType> & b)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(solveSystem);
    services::Status status;

    math::UpLo uplo = math::UpLo::Upper;

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(solveSystem.xpotrf);
        /* Perform L*L' decomposition of X'*X */
        status = LapackGpu<algorithmFPType>::xpotrf(uplo, p, a, p);
    }
    DAAL_CHECK_STATUS_VAR(status);

    {
        DAAL_ITTNOTIFY_SCOPED_TASK(solveSystem.xpotrs);
        /* Solve L*L'*b=Y */
        status = LapackGpu<algorithmFPType>::xpotrs(uplo, p, ny, a, p, b, p);
    }
    DAAL_CHECK_STATUS_VAR(status);
    return status;
}

} // namespace internal
} // namespace training
} // namespace normal_equations
} // namespace linear_model
} // namespace algorithms
} // namespace daal
