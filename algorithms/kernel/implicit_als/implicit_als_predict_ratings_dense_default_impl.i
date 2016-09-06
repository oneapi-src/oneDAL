/* file: implicit_als_predict_ratings_dense_default_impl.i */
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
//  Implementation of impicit ALS prediction algorithm
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_IMPL_I__
#define __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_IMPL_I__

#include "implicit_als_predict_ratings_dense_default_kernel.h"
#include "service_micro_table.h"
#include "service_sort.h"
#include "service_blas.h"
#include "threading.h"

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
void ImplicitALSPredictKernel<algorithmFPType, cpu>::getFactors(
            daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> &mtFactors1,
            size_t nRows1, algorithmFPType **factors1,
            daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> &mtFactors2,
            size_t nRows2, algorithmFPType **factors2)
{
    size_t nRowsRead = mtFactors1.getBlockOfRows(0, nRows1, factors1);
    if (nRowsRead < nRows1)
    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    nRowsRead = mtFactors2.getBlockOfRows(0, nRows2, factors2);
    if (nRowsRead < nRows2)
    {
        this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
        mtFactors1.release();
        return;
    }
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSPredictKernel<algorithmFPType, cpu>::compute(
            const NumericTable *usersFactorsTable, const NumericTable *itemsFactorsTable,
            NumericTable *ratingsTable, const Parameter *parameter)
{
    daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> mtUsersFactors(usersFactorsTable);
    daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> mtItemsFactors(itemsFactorsTable);

    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> mtRatings(ratingsTable);

    size_t nUsers = mtUsersFactors.getFullNumberOfRows();
    size_t nItems = mtItemsFactors.getFullNumberOfRows();

    algorithmFPType *usersFactors, *itemsFactors;
    getFactors(mtUsersFactors, nUsers, &usersFactors, mtItemsFactors, nItems, &itemsFactors);
    if (!this->_errors->isEmpty()) { return; }

    algorithmFPType *ratings;
    size_t nRowsRead = mtRatings.getBlockOfRows(0, nUsers, &ratings);
    if (nRowsRead < nUsers)
    {
        this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
        mtUsersFactors.release();
        mtItemsFactors.release();
        return;
    }

    size_t nFactors = parameter->nFactors;

    /* GEMM parameters */
    char trans   = 'T';
    char notrans = 'N';
    algorithmFPType one  = 1.0;
    algorithmFPType zero = 0.0;

    Blas<algorithmFPType, cpu>::xgemm(&trans, &notrans, (MKL_INT *)&nItems, (MKL_INT *)&nUsers, (MKL_INT *)&nFactors,
                       &one, itemsFactors, (MKL_INT *)&nFactors, usersFactors, (MKL_INT *)&nFactors, &zero,
                       ratings, (MKL_INT *)&nItems);

    mtRatings.release();
    mtUsersFactors.release();
    mtItemsFactors.release();
}

}
}
}
}
}
}

#endif
