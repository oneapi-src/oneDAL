/* file: implicit_als_train_init_csr_default_distr_impl.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of impicit ALS model initialization in distributed mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_DISTR_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_CSR_DEFAULT_DISTR_IMPL_I__

#include "service_micro_table.h"
#include "service_memory.h"
#include "service_rng.h"
#include "service_spblas.h"
#include "implicit_als_train_dense_default_batch_common.i"

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
template <typename algorithmFPType, CpuType cpu>
void ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu>::compute(
            const NumericTable *data, NumericTable *itemsFactorsTable, const Parameter *parameter)
{
    daal::internal::CSRBlockMicroTable<algorithmFPType, readOnly, cpu> mtData(data);
    size_t nItems = mtData.getFullNumberOfRows();
    size_t nUsers = mtData.getFullNumberOfColumns();
    size_t nFactors = parameter->nFactors;
    algorithmFPType *tdata;
    size_t *rowIndices, *colOffsets;

    size_t nItemsRead = mtData.getSparseBlock(0, nItems, &tdata, &rowIndices, &colOffsets);
    if (nItemsRead < nItems)
    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }


    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> partialFactors(itemsFactorsTable);
    algorithmFPType *itemsFactors;
    nItemsRead = partialFactors.getBlockOfRows(0, nItems, &itemsFactors);
    if (nItemsRead < nItems)
    {
        this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
        mtData.release();
        return;
    }

    size_t seed = parameter->seed;
    computePartialFactors(nUsers, nItems, nFactors, parameter->fullNUsers, seed,
                          tdata, rowIndices, colOffsets, itemsFactors);
    mtData.release();

    daal::internal::IntRng<int,cpu> rng(seed);

    algorithmFPType *randBuffer = (algorithmFPType *)daal::services::daal_malloc(nFactors * sizeof(algorithmFPType));
    if (!randBuffer)
    {
        this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
        return;
    }

    for (size_t i = 0; i < nItems; i++)
    {
        this->randFactors(1, nFactors, itemsFactors + i * nFactors, (int *)randBuffer, rng);
    }

    partialFactors.release();
    daal::services::daal_free(randBuffer);
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSInitDistrKernel<algorithmFPType, fastCSR, cpu>::computePartialFactors(
            size_t nUsers, size_t nItems, size_t nFactors, size_t fullNUsers, size_t seed,
            algorithmFPType *tdata, size_t *rowIndices, size_t *colOffsets, algorithmFPType *partialFactors)
{
    algorithmFPType one = 1.0;
    size_t bufSz = (nItems > nFactors ? nItems : nFactors);
    algorithmFPType *ones = (algorithmFPType *)daal::services::daal_malloc(nUsers * sizeof(algorithmFPType));
    algorithmFPType *itemsSum = (algorithmFPType *)daal::services::daal_malloc(bufSz * sizeof(algorithmFPType));
    if (!ones || !itemsSum)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    service_memset<algorithmFPType, cpu>(ones, one, nUsers);

    /* Parameters of CSRMV function */
    char transa = 'N';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 0.0;
    char matdescra[6] = {'\0', '\0', '\0', '\0', '\0', '\0'};
    matdescra[0] = 'G';        // general matrix
    matdescra[3] = 'F';        // 1-based indexing

    /* Compute sum of columns of input matrix */
    SpBlas<algorithmFPType, cpu>::xcsrmv(&transa, (MKL_INT *)&nItems, (MKL_INT *)&nUsers, &alpha, matdescra,
                        tdata, (MKL_INT *)rowIndices, (MKL_INT *)colOffsets, (MKL_INT *)(colOffsets + 1),
                        ones, &beta, itemsSum);
    daal::services::daal_free(ones);

    algorithmFPType invFullNUsers = one / (algorithmFPType)fullNUsers;
    for (size_t i = 0; i < nItems; i++)
    {
        partialFactors[i * nFactors] = itemsSum[i] * invFullNUsers;
    }

    daal::services::daal_free(itemsSum);
}

}
}
}
}
}
}

#endif
