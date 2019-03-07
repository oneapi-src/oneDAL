/* file: implicit_als_predict_ratings_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of impicit ALS prediction algorithm
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_IMPL_I__
#define __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_IMPL_I__

#include "implicit_als_predict_ratings_dense_default_kernel.h"
#include "service_numeric_table.h"
#include "service_blas.h"

using namespace daal::data_management;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSPredictKernel<algorithmFPType, cpu>::compute(
            const NumericTable *usersFactorsTable, const NumericTable *itemsFactorsTable,
            NumericTable *ratingsTable, const Parameter *parameter)
{
    const size_t nUsers = usersFactorsTable->getNumberOfRows();
    const size_t nItems = itemsFactorsTable->getNumberOfRows();

    ReadRows<algorithmFPType, cpu> mtUsersFactors(*const_cast<NumericTable*>(usersFactorsTable), 0, nUsers);
    DAAL_CHECK_BLOCK_STATUS(mtUsersFactors);
    ReadRows<algorithmFPType, cpu> mtItemsFactors(*const_cast<NumericTable*>(itemsFactorsTable), 0, nItems);
    DAAL_CHECK_BLOCK_STATUS(mtItemsFactors);
    WriteOnlyRows<algorithmFPType, cpu> mtRatings(*ratingsTable, 0, nUsers);
    DAAL_CHECK_BLOCK_STATUS(mtRatings);


    const algorithmFPType *usersFactors = mtUsersFactors.get();
    const algorithmFPType *itemsFactors = mtItemsFactors.get();
    algorithmFPType *ratings = mtRatings.get();
    const size_t nFactors = parameter->nFactors;

    /* GEMM parameters */
    const char trans   = 'T';
    const char notrans = 'N';
    const algorithmFPType one(1.0);
    const algorithmFPType zero(0.0);

    Blas<algorithmFPType, cpu>::xgemm(&trans, &notrans, (DAAL_INT *)&nItems, (DAAL_INT *)&nUsers, (DAAL_INT *)&nFactors,
                       &one, itemsFactors, (DAAL_INT *)&nFactors, usersFactors, (DAAL_INT *)&nFactors, &zero,
                       ratings, (DAAL_INT *)&nItems);
    return services::Status();
}

}
}
}
}
}
}

#endif
