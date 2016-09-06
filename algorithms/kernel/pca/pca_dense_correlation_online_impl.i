/* file: pca_dense_correlation_online_impl.i */
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
void PCACorrelationKernel<online, algorithmFPType, cpu>::compute(
    const data_management::NumericTablePtr data,
    PartialResult<correlationDense> *partialResult,
    OnlineParameter<algorithmFPType, correlationDense> *parameter)
{
    parameter->covariance->input.set(covariance::data, data);
    parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;

    parameter->covariance->computeNoThrow();
    if(parameter->covariance->getErrors()->size() != 0) {this->_errors->add(parameter->covariance->getErrors()->getErrors()); return;}

    copyCovarianceResultToPartialResult(parameter->covariance->getPartialResult().get(), partialResult);
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationKernel<online, algorithmFPType, cpu>::finalize(
    PartialResult<correlationDense> *partialResult,
    OnlineParameter<algorithmFPType, correlationDense> *parameter,
    data_management::NumericTablePtr eigenvectors,
    data_management::NumericTablePtr eigenvalues)
{
    parameter->covariance->finalizeCompute();
    this->_errors->add(parameter->covariance->getErrors()->getErrors());

    data_management::NumericTablePtr correlation = parameter->covariance->getResult()->get(covariance::covariance);

    this->computeCorrelationEigenvalues(correlation, eigenvectors, eigenvalues);
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationKernel<online, algorithmFPType, cpu>::copyCovarianceResultToPartialResult(
    covariance::PartialResult *covariancePres,
    PartialResult<correlationDense> *partialResult)
{
    copyIfNeeded(covariancePres->get(covariance::sum).get()          , partialResult->get(sumCorrelation).get());
    copyIfNeeded(covariancePres->get(covariance::nObservations).get(), partialResult->get(nObservationsCorrelation).get());
    copyIfNeeded(covariancePres->get(covariance::crossProduct).get() , partialResult->get(crossProductCorrelation).get());
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationKernel<online, algorithmFPType, cpu>::copyIfNeeded(data_management::NumericTable *src, data_management::NumericTable *dst)
{
    if(src == dst)
    {
        return;
    }
    else
    {
        size_t nRows = dst->getNumberOfRows();
        size_t nCols = dst->getNumberOfColumns();

        BlockDescriptor<algorithmFPType> srcBlock;
        src->getBlockOfRows(0, nRows, readOnly, srcBlock);
        algorithmFPType *srcArray = srcBlock.getBlockPtr();

        BlockDescriptor<algorithmFPType> dstBlock;
        dst->getBlockOfRows(0, nRows, writeOnly, dstBlock);
        algorithmFPType *dstArray = dstBlock.getBlockPtr();

        size_t nDataElements = nRows * nCols;
        daal::services::daal_memcpy_s(dstArray, nDataElements * sizeof(algorithmFPType), srcArray, nDataElements * sizeof(algorithmFPType));

        src->releaseBlockOfRows(srcBlock);
        dst->releaseBlockOfRows(dstBlock);
    }
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
