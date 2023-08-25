/* file: implicit_als_train_init_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_blas.h"
#include "src/algorithms/implicit_als/implicit_als_train_init_kernel.h"

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
services::Status ImplicitALSInitKernel<algorithmFPType, defaultDense, cpu>::compute(const NumericTable * dataTable, NumericTable * itemsFactorsTable,
                                                                                    NumericTable * usersFactorsTable, const Parameter * parameter,
                                                                                    engines::BatchBase & engine)
{
    const size_t nUsers   = dataTable->getNumberOfRows();
    const size_t nItems   = dataTable->getNumberOfColumns();
    const size_t nFactors = parameter->nFactors;

    const size_t bufSz = (nItems > nFactors ? nItems : nFactors);

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nUsers, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, bufSz, sizeof(algorithmFPType));

    TArray<algorithmFPType, cpu> ones(nUsers);
    TArray<algorithmFPType, cpu> itemsSum(bufSz);
    DAAL_CHECK_MALLOC(ones.get() && itemsSum.get());

    {
        ReadRows<algorithmFPType, cpu> mtData(*const_cast<NumericTable *>(dataTable), 0, nUsers);
        DAAL_CHECK_BLOCK_STATUS(mtData);
        const algorithmFPType * data = mtData.get();
        const algorithmFPType one(1.0);
        service_memset<algorithmFPType, cpu>(ones.get(), one, nUsers);
        /* Parameters of GEMV function */
        char transa           = 'N';
        algorithmFPType alpha = 1.0;
        algorithmFPType beta  = 0.0;
        DAAL_INT ione         = 1;

        /* Compute sum of rows of input matrix */
        BlasInst<algorithmFPType, cpu>::xgemv(&transa, (DAAL_INT *)&nItems, (DAAL_INT *)&nUsers, &alpha, const_cast<algorithmFPType *>(data),
                                              (DAAL_INT *)&nItems, ones.get(), (DAAL_INT *)&ione, &beta, itemsSum.get(), &ione);
    }

    WriteOnlyRows<algorithmFPType, cpu> mtItemsFactors(itemsFactorsTable, 0, nItems);
    DAAL_CHECK_BLOCK_STATUS(mtItemsFactors);
    algorithmFPType * itemsFactors = mtItemsFactors.get();

    DAAL_CHECK_STATUS_VAR(this->randFactors(nItems, nFactors, itemsFactors, engine));

    const algorithmFPType invNUsers = algorithmFPType(1.0) / algorithmFPType(nUsers);
    for (size_t i = 0; i < nItems; i++)
    {
        itemsFactors[i * nFactors] = itemsSum[i] * invNUsers;
    }

    return services::Status();
}

} // namespace internal
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
