/* file: pca_dense_correlation_distr_step2_impl.i */
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
//  Functuons that are used in PCA algorithm
//--
*/

#ifndef __PCA_DENSE_CORRELATION_DISTR_STEP2_IMPL_I__
#define __PCA_DENSE_CORRELATION_DISTR_STEP2_IMPL_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "pca_dense_correlation_distr_step2_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<distributed, algorithmFPType, cpu>::compute(DistributedInput<correlationDense> *input,
                                                                      PartialResult<correlationDense> *partialResult,
    const DistributedParameter<step2Master, algorithmFPType, correlationDense> *parameter)
{
    for(size_t i = 0; i < input->get(partialResults)->size(); i++)
    {
        covariance::PartialResultPtr covariancePartialResult(new covariance::PartialResult());
        covariancePartialResult->set(covariance::nObservations, input->getPartialResult(i)->get(pca::nObservationsCorrelation));
        covariancePartialResult->set(covariance::crossProduct, input->getPartialResult(i)->get(pca::crossProductCorrelation));
        covariancePartialResult->set(covariance::sum, input->getPartialResult(i)->get(pca::sumCorrelation));
        parameter->covariance->input.add(covariance::partialResults, covariancePartialResult);
    }
    parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;
    return parameter->covariance->computeNoThrow();
}

template <typename algorithmFPType, CpuType cpu>
services::Status PCACorrelationKernel<distributed, algorithmFPType, cpu>::finalize(PartialResult<correlationDense> *partialResult,
    const DistributedParameter<step2Master, algorithmFPType, correlationDense> *parameter,
    data_management::NumericTable& eigenvectors, data_management::NumericTable& eigenvalues)
{
    parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;
    services::Status s = parameter->covariance->finalizeCompute();
    if(!s)
        return s;

    data_management::NumericTablePtr correlation = parameter->covariance->getResult()->get(covariance::covariance);
    return this->computeCorrelationEigenvalues(*correlation, eigenvectors, eigenvalues);
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
