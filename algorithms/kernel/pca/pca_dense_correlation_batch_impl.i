/* file: pca_dense_correlation_batch_impl.i */
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
//  Functuons that are used in PCA algorithm
//--
*/

#ifndef __PCA_DENSE_CORRELATION_BATCH_IMPL_I__
#define __PCA_DENSE_CORRELATION_BATCH_IMPL_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"
#include "threading.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

using namespace daal::services::internal;
using namespace daal::data_management;
using namespace daal::internal;
template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<batch, algorithmFPType, cpu>::compute
        (bool isCorrelation,
         const data_management::NumericTable& dataTable,
         covariance::BatchImpl* covarianceAlg,
         data_management::NumericTable& eigenvectors,
         data_management::NumericTable& eigenvalues)
{
    if(isCorrelation)
        return this->computeCorrelationEigenvalues(dataTable, eigenvectors, eigenvalues);
    DAAL_CHECK(covarianceAlg, services::ErrorNullPtr);
    services::Status status;
    covarianceAlg->parameter.outputMatrixType = covariance::correlationMatrix;

    DAAL_CHECK_STATUS(status, covarianceAlg->computeNoThrow());
    return this->computeCorrelationEigenvalues(*covarianceAlg->getResult()->get(covariance::covariance), eigenvectors, eigenvalues);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<batch, algorithmFPType, cpu>::compute
        (bool isCorrelation,
         bool isDeterministic,
         const data_management::NumericTable& dataTable,
         covariance::BatchImpl* covarianceAlg,
         DAAL_UINT64 resultsToCompute,
         data_management::NumericTable& eigenvectors,
         data_management::NumericTable& eigenvalues,
         data_management::NumericTable& means,
         data_management::NumericTable& variances)
{
    services::Status status;
    if (isCorrelation)
    {
        if (resultsToCompute & mean)
        {
            DAAL_CHECK_STATUS(status, this->fillTable(means, (algorithmFPType)0));
        }

        if (resultsToCompute & variance)
        {
            DAAL_CHECK_STATUS(status, this->fillTable(variances, (algorithmFPType)1));
        }
        DAAL_CHECK_STATUS(status, this->computeCorrelationEigenvalues(dataTable, eigenvectors, eigenvalues));
    }
    else
    {
        DAAL_CHECK(covarianceAlg, services::ErrorNullPtr);
        DAAL_CHECK_STATUS(status, covarianceAlg->computeNoThrow());
        auto pCovarianceTable = covarianceAlg->getResult()->get(covariance::covariance);
        NumericTable& covarianceTable = *pCovarianceTable;
        if (resultsToCompute & mean)
        {
            DAAL_CHECK_STATUS(status, this->copyTable(*covarianceAlg->getResult()->get(covariance::mean), means));
        }

        if (resultsToCompute & variance)
        {
            DAAL_CHECK_STATUS(status, this->copyVarianceFromCovarianceTable(covarianceTable, variances));
        }

        DAAL_CHECK_STATUS(status, this->correlationFromCovarianceTable(covarianceTable));
        DAAL_CHECK_STATUS(status, this->computeCorrelationEigenvalues(covarianceTable, eigenvectors, eigenvalues));
    }

    if (isDeterministic)
    {
        DAAL_CHECK_STATUS(status, this->signFlipEigenvectors(eigenvectors));
    }

    return status;
}


} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
