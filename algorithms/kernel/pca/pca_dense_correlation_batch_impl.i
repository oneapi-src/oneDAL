/* file: pca_dense_correlation_batch_impl.i */
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

#ifndef __PCA_DENSE_CORRELATION_BATCH_IMPL_I__
#define __PCA_DENSE_CORRELATION_BATCH_IMPL_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_numeric_table.h"
#include "pca_dense_correlation_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
void PCACorrelationKernel<batch, algorithmFPType, cpu>::compute(const data_management::NumericTablePtr data,
                                                                const bool isCorrelation,
                                                                BatchParameter<algorithmFPType, correlationDense> *parameter,
                                                                data_management::NumericTablePtr eigenvectors,
                                                                data_management::NumericTablePtr eigenvalues)
{
    data_management::NumericTablePtr correlation;
    if(!isCorrelation)
    {
        parameter->covariance->input.set(covariance::data, data);
        parameter->covariance->parameter.outputMatrixType = covariance::correlationMatrix;

        parameter->covariance->computeNoThrow();
        if(parameter->covariance->getErrors()->size() != 0) {this->_errors->add(parameter->covariance->getErrors()->getErrors()); return;}

        correlation = parameter->covariance->getResult()->get(covariance::covariance);
    }
    else
    {
        correlation = data;
    }

    this->computeCorrelationEigenvalues(correlation, eigenvectors, eigenvalues);
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
