/* file: covariance_dense_batch_impl.i */
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
//  Covariance matrix computation algorithm implementation in batch mode
//--
*/

#ifndef __COVARIANCE_CSR_BATCH_IMPL_I__
#define __COVARIANCE_CSR_BATCH_IMPL_I__

#include "covariance_kernel.h"
#include "covariance_impl.i"

#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDenseBatchKernel<algorithmFPType, method, cpu>::compute(
    NumericTable *dataTable,
    NumericTable *covTable,
    NumericTable *meanTable,
    const Parameter *parameter)
{
    algorithmFPType nObservations = 0.0;

    const size_t nFeatures  = dataTable->getNumberOfColumns();
    const size_t nVectors   = dataTable->getNumberOfRows();
    const bool isNormalized = dataTable->isNormalized(NumericTableIface::standardScoreNormalized);

    DEFINE_TABLE_BLOCK( ReadRows,      dataBlock,          dataTable );
    DEFINE_TABLE_BLOCK( WriteOnlyRows, sumBlock,           meanTable );
    DEFINE_TABLE_BLOCK( WriteOnlyRows, crossProductBlock,  covTable  );

    algorithmFPType *sums          = sumBlock.get();
    algorithmFPType *crossProduct  = crossProductBlock.get();
    algorithmFPType *data          = const_cast<algorithmFPType *>(dataBlock.get());

    services::Status status;

    status |= prepareSums<algorithmFPType, method, cpu>(dataTable, sums);
    DAAL_CHECK_STATUS_VAR(status);

    status |= prepareCrossProduct<algorithmFPType, cpu>(nFeatures, crossProduct);
    DAAL_CHECK_STATUS_VAR(status);

    status |= updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(isNormalized, nFeatures,
        nVectors, data, crossProduct, sums, &nObservations);
    DAAL_CHECK_STATUS_VAR(status);

    status |= finalizeCovariance<algorithmFPType, cpu>(
        nFeatures, nObservations, crossProduct, sums, crossProduct, sums, parameter);

    return status;
}

} // internal
} // covariance
} // algorithms
} // daal

#endif
