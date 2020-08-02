/* file: svm_regression_train_thunder_impl.i */
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

#ifndef __SVM_REGRESSION_TRAIN_THUNDER_IMPL_I__
#define __SVM_REGRESSION_TRAIN_THUNDER_IMPL_I__

#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"
#include "src/externals/service_ittnotify.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_math.h"

#include "src/algorithms/svm/svm_train_common.h"
#include "src/algorithms/svm/svm_train_thunder_workset.h"
#include "src/algorithms/svm/svm_train_thunder_cache.h"
#include "src/algorithms/svm/svm_train_result.h"

#include "src/algorithms/svm/svm_train_common_impl.i"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace classification
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status SVMTrainImpl<thunder, algorithmFPTypem cpu>::compute(const NumericTablePtr & xTable, const NumericTablePtr & wTable,
                                                                      NumericTable & yTable, daal::algorithms::Model * r,
                                                                      const ParameterType * svmPar)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(COMPUTE);

    services::Status status;

    const algorithmFPType C(svmPar->C);
    const algorithmFPType epsilon(svmPar->epsilon);
    const algorithmFPType accuracyThreshold(svmPar->accuracyThreshold);
    const algorithmFPType tau(svmPar->tau);
    const size_t maxIterations(svmPar->maxIterations);
    const size_t cacheSize(svmPar->cacheSize);
    kernel_function::KernelIfacePtr kernel = svmPar->kernel->clone();

    const size_t nVectors    = xTable->getNumberOfRows();
    const size_t nTwoVectors = nVectors * 2;

    TArray<algorithmFPType, cpu> alphaTArray(nTwoVectors);
    DAAL_CHECK_MALLOC(alphaTArray.get());
    algorithmFPType * const alpha = alphaTArray.get();

    TArray<algorithmFPType, cpu> gradTArray(nTwoVectors);
    DAAL_CHECK_MALLOC(gradTArray.get());
    algorithmFPType * const grad = gradTArray.get();

    TArray<algorithmFPType, cpu> cwTArray(nTwoVectors);
    DAAL_CHECK_MALLOC(cwTArray.get());
    algorithmFPType * const cw = cwTArray.get();

    TArray<algorithmFPType, cpu> yTArray(nTwoVectors);
    DAAL_CHECK_MALLOC(yTArray.get());
    algorithmFPType * const y = yTArray.get();

    SafeStatus safeStat;

    size_t nNonZeroWeights = nTwoVectors;
    {
        /* The operation copy is lightweight, therefore a large size is chosen
            so that the number of blocks is a reasonable number. */
        const size_t blockSize = 16384;
        const size_t nBlocks   = nVectors / blockSize + !!(nVectors % blockSize);

        TlsSum<size_t, cpu> weightsCounter(1);
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            const size_t startRow     = iBlock * blockSize;
            const size_t nRowsInBlock = (iBlock != nBlocks - 1) ? blockSize : nVectors - iBlock * blockSize;

            ReadColumns<algorithmFPType, cpu> mtY(yTable, 0, startRow, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(mtY);
            const algorithmFPType * const yIn = mtY.get();

            ReadColumns<algorithmFPType, cpu> mtW(wTable.get(), 0, startRow, nRowsInBlock);
            DAAL_CHECK_BLOCK_STATUS_THR(mtW);
            const algorithmFPType * weights = mtW.get();

            size_t * wc = nullptr;
            if (weights)
            {
                wc = weightsCounter.local();
            }
            for (size_t i = 0; i < nRowsInBlock; ++i)
            {
                y[i + startRow]                = algorithmFPType(1.0);
                y[nVectors + i + startRow]     = algorithmFPType(-1.0);
                grad[i + startRow]             = epsilon - y[i + startRow];
                grad[nVectors + i + startRow]  = -epsilon - y[i + startRow];
                alpha[i + startRow]            = algorithmFPType(0);
                alpha[nVectors + i + startRow] = algorithmFPType(0);
                cw[i + startRow]               = weights ? weights[i] * C : C;
                cw[nVectors + i + startRow]    = weights ? weights[i] * C : C;
                if (weights)
                {
                    *wc += static_cast<size_t>(weights[i] != algorithmFPType(0));
                }
            }
        });

        if (wTable.get())
        {
            weightsCounter.reduceTo(&nNonZeroWeights, 1);
        }
    }

    internal::Solver<thunder, algorithmFPType, cpu> solver(xTable, nNonZeroWeights, cw, y alpha, grad);

    DAAL_CHECK_STATUS(status, solver.compute());

    printf("IS DONE\n");

    return status;
}

} // namespace internal
} // namespace training
} // namespace classification
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
