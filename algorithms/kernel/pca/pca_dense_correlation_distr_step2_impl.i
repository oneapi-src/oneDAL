/* file: pca_dense_correlation_distr_step2_impl.i */
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
void PCACorrelationKernel<distributed, algorithmFPType, cpu>::compute(DistributedInput<correlationDense> *input,
                                                                      PartialResult<correlationDense> *partialResult,
                                                                      DistributedParameter<step2Master, algorithmFPType, correlationDense> *parameter)
{
    for(size_t i = 0; i < input->get(partialResults)->size(); i++)
    {
        services::SharedPtr<covariance::PartialResult> covariancePartialResult = services::SharedPtr<covariance::PartialResult>
                                                                                 (new covariance::PartialResult());
        covariancePartialResult->set(covariance::nObservations, input->getPartialResult(i)->get(pca::nObservationsCorrelation));
        covariancePartialResult->set(covariance::crossProduct, input->getPartialResult(i)->get(pca::crossProductCorrelation));
        covariancePartialResult->set(covariance::sum, input->getPartialResult(i)->get(pca::sumCorrelation));
        parameter->covariance->input.add(covariance::partialResults, covariancePartialResult);
    }

    parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;

    parameter->covariance->computeNoThrow();
    if(parameter->covariance->getErrors()->size() != 0) {this->_errors->add(parameter->covariance->getErrors()->getErrors()); return;}
}

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationKernel<distributed, algorithmFPType, cpu>::finalize(PartialResult<correlationDense> *partialResult,
                                                                       DistributedParameter<step2Master,
                                                                       algorithmFPType, correlationDense> *parameter,
                                                                       data_management::NumericTablePtr eigenvectors,
                                                                       data_management::NumericTablePtr eigenvalues)
{
    parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;

    parameter->covariance->finalizeCompute();
    this->_errors->add(parameter->covariance->getErrors()->getErrors());

    data_management::NumericTablePtr correlation = parameter->covariance->getResult()->get(covariance::covariance);

    this->computeCorrelationEigenvalues(correlation, eigenvectors, eigenvalues);
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
