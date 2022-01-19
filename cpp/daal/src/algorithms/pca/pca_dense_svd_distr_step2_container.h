/* file: pca_dense_svd_distr_step2_container.h */
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

#ifndef __PCA_DENSE_SVD_DISTR_STEP2_CONTAINER_H__
#define __PCA_DENSE_SVD_DISTR_STEP2_CONTAINER_H__

#include "src/algorithms/kernel.h"
#include "algorithms/pca/pca_distributed.h"
#include "src/algorithms/pca/pca_dense_svd_distr_step2_kernel.h"
#include "src/algorithms/pca/pca_dense_svd_container.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
template <typename algorithmFPType, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, svdDense, cpu>::DistributedContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::PCASVDStep2MasterKernel, algorithmFPType);
}

template <typename algorithmFPType, CpuType cpu>
DistributedContainer<step2Master, algorithmFPType, svdDense, cpu>::~DistributedContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, svdDense, cpu>::compute()
{
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status DistributedContainer<step2Master, algorithmFPType, svdDense, cpu>::finalizeCompute()
{
    Result * result = static_cast<Result *>(_res);

    DistributedInput<svdDense> * input = static_cast<DistributedInput<svdDense> *>(_in);

    data_management::DataCollectionPtr inputPartialResults = input->get(pca::partialResults);

    data_management::NumericTablePtr eigenvalues  = result->get(pca::eigenvalues);
    data_management::NumericTablePtr eigenvectors = result->get(pca::eigenvectors);

    daal::services::Environment::env & env = *_env;

    Status s = __DAAL_CALL_KERNEL_STATUS(env, internal::PCASVDStep2MasterKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType), finalizeMerge,
                                         internal::nonNormalizedDataset, inputPartialResults, *eigenvalues, *eigenvectors);

    inputPartialResults->clear();
    return s;
}

} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
