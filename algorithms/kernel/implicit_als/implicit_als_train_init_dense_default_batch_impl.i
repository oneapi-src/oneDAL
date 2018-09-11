/* file: implicit_als_train_init_dense_default_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of defaultDense method for impicit ALS initialization
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_INIT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_INIT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "service_numeric_table.h"
#include "service_memory.h"
#include "service_blas.h"
#include "implicit_als_train_init_kernel.h"

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
services::Status ImplicitALSInitKernel<algorithmFPType, defaultDense, cpu>::compute(
                 const NumericTable *dataTable, NumericTable *itemsFactorsTable, const Parameter *parameter, engines::BatchBase &engine)
{
    const size_t nUsers = dataTable->getNumberOfRows();
    const size_t nItems = dataTable->getNumberOfColumns();
    const size_t nFactors = parameter->nFactors;

    const size_t bufSz = (nItems > nFactors ? nItems : nFactors);
    TArray<algorithmFPType, cpu> ones(nUsers);
    TArray<algorithmFPType, cpu> itemsSum(bufSz);
    DAAL_CHECK_MALLOC(ones.get() && itemsSum.get());

    {
        ReadRows<algorithmFPType, cpu> mtData(*const_cast<NumericTable*>(dataTable), 0, nUsers);
        DAAL_CHECK_BLOCK_STATUS(mtData);
        const algorithmFPType *data = mtData.get();
        const algorithmFPType one(1.0);
        service_memset<algorithmFPType, cpu>(ones.get(), one, nUsers);
    /* Parameters of GEMV function */
    char transa = 'N';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 0.0;
        DAAL_INT ione = 1;

    /* Compute sum of rows of input matrix */
        Blas<algorithmFPType, cpu>::xgemv(&transa, (DAAL_INT *)&nItems, (DAAL_INT *)&nUsers, &alpha,
            const_cast<algorithmFPType*>(data), (DAAL_INT *)&nItems,
            ones.get(), (DAAL_INT *)&ione, &beta, itemsSum.get(), &ione);
    }

    WriteOnlyRows<algorithmFPType, cpu> mtItemsFactors(itemsFactorsTable, 0, nItems);
    DAAL_CHECK_BLOCK_STATUS(mtItemsFactors);
    algorithmFPType *itemsFactors = mtItemsFactors.get();

    const algorithmFPType invNUsers = algorithmFPType(1.0) / algorithmFPType(nUsers);
    for (size_t i = 0; i < nItems; i++)
    {
        itemsFactors[i * nFactors] = itemsSum[i] * invNUsers;
    }

    return this->randFactors(nItems, nFactors, itemsFactors, (int*)itemsSum.get(), engine);
    //reusing itemsSum as an array of ints, to save on memory allocation
}

}
}
}
}
}
}

#endif
