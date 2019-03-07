/* file: pca_dense_correlation_online_impl.i */
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

#ifndef __PCA_DENSE_CORRELATION_ONLINE_IMPL_I__
#define __PCA_DENSE_CORRELATION_ONLINE_IMPL_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "pca_dense_correlation_online_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<online, algorithmFPType, cpu>::compute(
    const data_management::NumericTablePtr& pData,
    PartialResult<correlationDense> *partialResult,
    const OnlineParameter<algorithmFPType, correlationDense> *parameter)
{
    parameter->covariance->input.set(covariance::data, pData);
    parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;

    services::Status s = parameter->covariance->computeNoThrow();
    if(s)
        s = copyCovarianceResultToPartialResult(parameter->covariance->getPartialResult().get(), partialResult);
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<online, algorithmFPType, cpu>::finalize(
    PartialResult<correlationDense> *partialResult,
    const OnlineParameter<algorithmFPType, correlationDense> *parameter,
    data_management::NumericTable& eigenvectors,
    data_management::NumericTable& eigenvalues)
{
    services::Status s = parameter->covariance->finalizeCompute();
    if(!s)
        return s;

    data_management::NumericTablePtr correlation = parameter->covariance->getResult()->get(covariance::covariance);
    return this->computeCorrelationEigenvalues(*correlation, eigenvectors, eigenvalues);
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<online, algorithmFPType, cpu>::copyCovarianceResultToPartialResult(
    covariance::PartialResult *covariancePres,
    PartialResult<correlationDense> *partialResult)
{
    services::Status s = copyIfNeeded(covariancePres->get(covariance::sum).get(), partialResult->get(sumCorrelation).get());
    s |= copyIfNeeded(covariancePres->get(covariance::nObservations).get(), partialResult->get(nObservationsCorrelation).get());
    s |= copyIfNeeded(covariancePres->get(covariance::crossProduct).get(), partialResult->get(crossProductCorrelation).get());
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<online, algorithmFPType, cpu>::copyIfNeeded(data_management::NumericTable *src, data_management::NumericTable *dst)
{
    if(src == dst)
        return services::Status();

    const size_t nRows = dst->getNumberOfRows();
    const size_t nCols = dst->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> srcBlock(*src, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(srcBlock);
    const algorithmFPType *srcArray = srcBlock.get();

    WriteOnlyRows<algorithmFPType, cpu> dstBlock(*dst, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(srcBlock);
    algorithmFPType *dstArray = dstBlock.get();

    const size_t nDataElements = nRows * nCols;
    daal::services::daal_memcpy_s(dstArray, nDataElements * sizeof(algorithmFPType), srcArray, nDataElements * sizeof(algorithmFPType));
    return services::Status();
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
