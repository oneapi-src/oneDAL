/* file: svd_dense_default_distr_step3_impl.i */
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
//  Implementation of svds
//--
*/

#ifndef __SVD_DENSE_DEFAULT_DISTR_STEP3_IMPL_I__
#define __SVD_DENSE_DEFAULT_DISTR_STEP3_IMPL_I__

#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/services/service_defines.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

#include "src/algorithms/svd/svd_dense_default_impl.i"

#include "src/threading/threading.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace internal
{
template <typename algorithmFPType, daal::algorithms::svd::Method method, CpuType cpu>
Status SVDDistributedStep3Kernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                        NumericTable * r[], const daal::algorithms::Parameter * par)
{
    size_t i, j;
    size_t nBlocks     = na / 2;
    size_t mCalculated = 0;

    ReadRows<algorithmFPType, cpu, NumericTable> Aux1iBlock;
    ReadRows<algorithmFPType, cpu, NumericTable> Aux3iBlock;
    WriteOnlyRows<algorithmFPType, cpu, NumericTable> QiBlock;

    for (size_t k = 0; k < nBlocks; k++)
    {
        NumericTable * ntAux1i = const_cast<NumericTable *>(a[k]);
        NumericTable * ntAux3i = const_cast<NumericTable *>(a[k + nBlocks]);

        size_t n = ntAux1i->getNumberOfColumns();
        size_t m = ntAux1i->getNumberOfRows();

        const algorithmFPType * Aux1i = Aux1iBlock.set(ntAux1i, 0, m); /* Aux1i = Qin[m][n] */
        DAAL_CHECK_BLOCK_STATUS(Aux1iBlock);

        const algorithmFPType * Aux3i = Aux3iBlock.set(ntAux3i, 0, n); /* Aux3i = Ri [n][n] */
        DAAL_CHECK_BLOCK_STATUS(Aux3iBlock);

        algorithmFPType * Qi = QiBlock.set(r[0], mCalculated, m); /* Qi [m][n] */
        DAAL_CHECK_BLOCK_STATUS(QiBlock);

        TArray<algorithmFPType, cpu> QiTPtr(n * m);
        TArray<algorithmFPType, cpu> Aux1iTPtr(n * m);
        TArray<algorithmFPType, cpu> Aux3iTPtr(n * n);
        algorithmFPType * QiT    = QiTPtr.get();
        algorithmFPType * Aux1iT = Aux1iTPtr.get();
        algorithmFPType * Aux3iT = Aux3iTPtr.get();

        DAAL_CHECK(QiT && Aux1iT && Aux3iT, ErrorMemoryAllocationFailed);

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < m; j++)
            {
                Aux1iT[i * m + j] = Aux1i[i + j * n];
            }
        }
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                Aux3iT[i * n + j] = Aux3i[i + j * n];
            }
        }

        DAAL_INT ldAux1i = m;
        DAAL_INT ldAux3i = n;
        DAAL_INT ldQi    = m;

        const auto ec = compute_gemm_on_one_node<algorithmFPType, cpu>(m, n, Aux1iT, ldAux1i, Aux3iT, ldAux3i, QiT, ldQi);
        if (!ec) return ec;

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < m; j++)
            {
                Qi[i + j * n] = QiT[i * m + j];
            }
        }

        mCalculated += m;
    }

    return Status();
}

} // namespace internal
} // namespace svd
} // namespace algorithms
} // namespace daal

#endif
