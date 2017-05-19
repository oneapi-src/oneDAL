/* file: implicit_als_train_init_csr_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of defaultDense method for impicit ALS initialization
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_BATCH_IMPL_I__

#include "service_memory.h"
#include "service_spblas.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{

using namespace daal::services::internal;
using namespace daal::internal;

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSInitKernel<algorithmFPType, fastCSR, cpu>::compute(
                 const NumericTable *dataTable, NumericTable *itemsFactorsTable, const Parameter *parameter)
{
    const size_t nUsers = dataTable->getNumberOfRows();
    const size_t nItems = dataTable->getNumberOfColumns();
    const size_t nFactors = parameter->nFactors;

    const size_t bufSz = (nItems > nFactors ? nItems : nFactors);
    TArray<algorithmFPType, cpu> ones(nUsers);
    TArray<algorithmFPType, cpu> itemsSum(bufSz);
    DAAL_CHECK_MALLOC(ones.get() && itemsSum.get());
    const algorithmFPType one(1.0);
    service_memset<algorithmFPType, cpu>(ones.get(), one, nUsers);

    {
        const CSRNumericTableIface* csrIface = dynamic_cast<const CSRNumericTableIface *>(dataTable);
        ReadRowsCSR<algorithmFPType, cpu> mtData(*const_cast<CSRNumericTableIface *>(csrIface), 0, nUsers);
        DAAL_CHECK_BLOCK_STATUS(mtData);
        const algorithmFPType *data = mtData.values();
        const size_t *colIndices = mtData.cols();
        const size_t *rowOffsets = mtData.rows();


    /* Parameters of CSRMV function */
    char transa = 'T';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 0.0;
    char matdescra[6];
    matdescra[0] = 'G';        // general matrix
    matdescra[3] = 'F';        // 1-based indexing

        matdescra[1] = (char)0;
        matdescra[2] = (char)0;
        matdescra[4] = (char)0;
        matdescra[5] = (char)0;

    /* Compute sum of rows of input matrix */
    SpBlas<algorithmFPType, cpu>::xcsrmv(&transa, (DAAL_INT *)&nUsers, (DAAL_INT *)&nItems, &alpha, matdescra,
                        data, (DAAL_INT *)colIndices, (DAAL_INT *)rowOffsets, (DAAL_INT *)(rowOffsets + 1),
            ones.get(), &beta, itemsSum.get());
    }

    WriteOnlyRows<algorithmFPType, cpu> mtItemsFactors(itemsFactorsTable, 0, nItems);
    DAAL_CHECK_BLOCK_STATUS(mtItemsFactors);
    algorithmFPType *itemsFactors = mtItemsFactors.get();
    const algorithmFPType invNUsers = one / algorithmFPType(nUsers);
    for (size_t i = 0; i < nItems; i++)
    {
        itemsFactors[i * nFactors] = itemsSum[i] * invNUsers;
    }
    BaseRNGs<cpu> baseRng(parameter->seed);
    return this->randFactors(nItems, nFactors, itemsFactors, (int *)itemsSum.get(), baseRng);
}

}
}
}
}
}
}

#endif
