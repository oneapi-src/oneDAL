/* file: pca_dense_correlation_batch_container_v2.h */
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_BATCH_CONTAINER_V2_H__
#define __PCA_DENSE_CORRELATION_BATCH_CONTAINER_V2_H__

#include "kernel.h"
#include "pca/inner/pca_batch_v2.h"
#include "pca_dense_correlation_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface2
{

template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, correlationDense, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCACorrelationKernel, batch, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, correlationDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status BatchContainer<algorithmFPType, correlationDense, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);
    interface2::BatchParameter<algorithmFPType, correlationDense> *parameter = static_cast<interface2::BatchParameter<algorithmFPType,
                                                                                           correlationDense> *>(_par);
    services::Environment::env &env = *_env;

    data_management::NumericTablePtr data = input->get(pca::data);
    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);
    data_management::NumericTablePtr means        = result->get(pca::means);
    data_management::NumericTablePtr variances    = result->get(pca::variances);

    auto covarianceAlgorithm = parameter->covariance;
    covarianceAlgorithm->input.set(covariance::data, data);

    if (parameter->resultsToCompute & mean)
    {
        covarianceAlgorithm->getResult()->set(covariance::mean, means);
    }

    __DAAL_CALL_KERNEL(env, internal::PCACorrelationKernel, __DAAL_KERNEL_ARGUMENTS(batch, algorithmFPType), compute,
                       input->isCorrelation(), parameter->isDeterministic, *data, covarianceAlgorithm.get(),
                       parameter->resultsToCompute, *eigenvectors, *eigenvalues, *means, *variances);
}

} // namespace interface2
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
