/* file: implicit_als_train_dense_default_batch_aux.i */
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
//  Auxiliary functions needed to train impicit ALS with fastCSR method
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_AUX_I__
#define __IMPLICIT_ALS_TRAIN_DENSE_DEFAULT_BATCH_AUX_I__

#include "implicit_als_train_utils.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::services;

template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTaskBase<algorithmFPType, cpu>::ImplicitALSTrainTaskBase(const NumericTable * dataTable, implicit_als::Model * model,
                                                                         const Parameter * parameter)
    : mtItemsFactors(*model->getItemsFactors(), 0, dataTable->getNumberOfColumns()),
      mtUsersFactors(*model->getUsersFactors(), 0, dataTable->getNumberOfRows()),
      nItems(dataTable->getNumberOfColumns()),
      nUsers(dataTable->getNumberOfRows()),
      xtx(parameter->nFactors * parameter->nFactors),
      nFactors(parameter->nFactors)
{}

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSTrainTaskBase<algorithmFPType, cpu>::init(const NumericTable * dataTable, implicit_als::Model * initModel,
                                                                      const Parameter * parameter)
{
    DAAL_CHECK_MALLOC(xtx.get());
    DAAL_CHECK_BLOCK_STATUS(mtItemsFactors);
    DAAL_CHECK_BLOCK_STATUS(mtUsersFactors);
    int result = 0;

    daal::internal::ReadRows<algorithmFPType, cpu> mtInitItemsFactors(*initModel->getItemsFactors(), 0, nItems);
    DAAL_CHECK_BLOCK_STATUS(mtInitItemsFactors);
    if (mtItemsFactors.get() != mtInitItemsFactors.get())
    {
        result = daal::services::internal::daal_memcpy_s(mtItemsFactors.get(), nItems * nFactors * sizeof(algorithmFPType), mtInitItemsFactors.get(),
                                                         nItems * nFactors * sizeof(algorithmFPType));
    }
    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>::ImplicitALSTrainTask(const NumericTable * dataTable, implicit_als::Model * model,
                                                                          const Parameter * parameter)
    : ImplicitALSTrainTaskBase<algorithmFPType, cpu>(dataTable, model, parameter)
{}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSTrainTask<algorithmFPType, fastCSR, cpu>::init(const NumericTable * dataTable, implicit_als::Model * initModel,
                                                                 const Parameter * parameter)
{
    Status s = super::init(dataTable, initModel, parameter);
    if (!s) return s;
    mtData.set(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(dataTable)), 0, nUsers);
    DAAL_CHECK_BLOCK_STATUS(mtData);
    const size_t nNonNull = mtData.rows()[nUsers] - mtData.rows()[0];
    tdata.reset(nNonNull);
    rowIndices.reset(nNonNull);
    colOffsets.reset(nUsers + 1);
    DAAL_CHECK_MALLOC(tdata.get() && rowIndices.get() && colOffsets.get());
    return csr2csc<algorithmFPType, cpu>(nUsers, nItems, mtData.values(), mtData.cols(), mtData.rows(), tdata.get(), rowIndices.get(),
                                         colOffsets.get());
}

template <typename algorithmFPType, CpuType cpu>
ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>::ImplicitALSTrainTask(const NumericTable * dataTable, implicit_als::Model * model,
                                                                               const Parameter * parameter)
    : ImplicitALSTrainTaskBase<algorithmFPType, cpu>(dataTable, model, parameter)
{}

template <typename algorithmFPType, CpuType cpu>
services::Status ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>::init(const NumericTable * dataTable, implicit_als::Model * initModel,
                                                                                const Parameter * parameter)
{
    Status s = super::init(dataTable, initModel, parameter);
    if (!s) return s;
    mtData.set(*const_cast<NumericTable *>(dataTable), 0, nUsers);
    DAAL_CHECK_BLOCK_STATUS(mtData);
    tdata.reset(nItems * nUsers);
    DAAL_CHECK_MALLOC(tdata.get());
    transpose(nUsers, nItems, mtData.get(), tdata.get());
    return s;
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainTask<algorithmFPType, defaultDense, cpu>::transpose(size_t nRows, size_t nCols, const algorithmFPType * data,
                                                                         algorithmFPType * tdata)
{
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            tdata[j * nRows + i] = data[i * nCols + j];
        }
    }
}

} // namespace internal
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
