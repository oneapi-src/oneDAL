/* file: pca_dense_svd_batch_container.h */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
//  Implementation of PCA Correlation algorithm container.
//--
*/

#ifndef __PCA_DENSE_SVD_BATCH_CONTAINER_H__
#define __PCA_DENSE_SVD_BATCH_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/pca/pca_batch.h"
#include "src/algorithms/pca/pca_dense_svd_batch_kernel.h"
#include "src/algorithms/pca/pca_dense_svd_container.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface3
{
template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, svdDense, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCASVDBatchKernel, algorithmFPType, interface3::BatchParameter<algorithmFPType, pca::svdDense>);
}

template <typename algorithmFPType, CpuType cpu>
BatchContainer<algorithmFPType, svdDense, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
Status BatchContainer<algorithmFPType, svdDense, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);
    interface3::BatchParameter<algorithmFPType, pca::svdDense> * parameter =
        static_cast<interface3::BatchParameter<algorithmFPType, pca::svdDense> *>(_par);

    internal::InputDataType dtype = getInputDataType(input);

    data_management::NumericTablePtr data         = input->get(pca::data);
    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);
    data_management::NumericTablePtr means        = result->get(pca::means);
    data_management::NumericTablePtr variances    = result->get(pca::variances);

    auto normalizationAlgorithm = parameter->normalization;
    normalizationAlgorithm->input.set(normalization::zscore::data, data);

    auto algParameter = &(normalizationAlgorithm->parameter());
    if (parameter->resultsToCompute & mean)
    {
        algParameter->resultsToCompute |= normalization::zscore::mean;
    }

    if (parameter->resultsToCompute & variance)
    {
        algParameter->resultsToCompute |= normalization::zscore::variance;
    }

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::PCASVDBatchKernel,
                       __DAAL_KERNEL_ARGUMENTS(algorithmFPType, interface3::BatchParameter<algorithmFPType, pca::svdDense>), compute, dtype, *data,
                       parameter, *eigenvalues, *eigenvectors, *means, *variances);
}

} // namespace interface3
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
