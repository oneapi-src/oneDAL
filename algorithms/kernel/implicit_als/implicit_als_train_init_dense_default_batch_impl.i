/* file: implicit_als_train_init_dense_default_batch_impl.i */
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
//  Implementation of defaultDense method for impicit ALS initialization
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "service_micro_table.h"
#include "service_memory.h"
#include "service_blas.h"

using namespace daal::services::internal;
using namespace daal::internal;

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
void ImplicitALSInitKernel<algorithmFPType, defaultDense, cpu>::compute(
            const NumericTable *dataTable, NumericTable *itemsFactorsTable, const Parameter *parameter)
{
    daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> mtData(dataTable);
    size_t nUsers = mtData.getFullNumberOfRows();
    size_t nItems = mtData.getFullNumberOfColumns();
    size_t nFactors = parameter->nFactors;
    algorithmFPType *data;

    size_t nUsersRead = mtData.getBlockOfRows(0, nUsers, &data);
    if (nUsersRead < nUsers)
    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    ImplicitALSInitTask<algorithmFPType, cpu> task(itemsFactorsTable, this);
    if (!this->_errors->isEmpty()) { return; }

    algorithmFPType *itemsFactors = task.itemsFactors;

    size_t bufSz = (nItems > nFactors ? nItems : nFactors);
    algorithmFPType one = 1.0;
    algorithmFPType *ones = (algorithmFPType *)daal::services::daal_malloc(nUsers * sizeof(algorithmFPType));
    algorithmFPType *itemsSum = (algorithmFPType *)daal::services::daal_malloc(bufSz * sizeof(algorithmFPType));
    if (!ones || !itemsSum)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        mtData.release();
        return;
    }

    service_memset<algorithmFPType, cpu>(ones, one, nUsers);

    /* Parameters of GEMV function */
    char transa = 'N';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 0.0;
    size_t ione = 1;

    /* Compute sum of rows of input matrix */
    Blas<algorithmFPType, cpu>::xgemv(&transa, (MKL_INT *)&nItems, (MKL_INT *)&nUsers, &alpha, data, (MKL_INT *)&nItems,
                       ones, (MKL_INT *)&ione, &beta, itemsSum, (MKL_INT *)&ione);
    mtData.release();
    daal::services::daal_free(ones);

    algorithmFPType invNUsers = one / (algorithmFPType)nUsers;
    for (size_t i = 0; i < nItems; i++)
    {
        itemsFactors[i * nFactors] = itemsSum[i] * invNUsers;
    }

    IntRng<int,cpu> rng(parameter->seed);
    this->randFactors(nItems, nFactors, itemsFactors, (int *)itemsSum, rng);

    daal::services::daal_free(itemsSum);
}

}
}
}
}
}
}

#endif
