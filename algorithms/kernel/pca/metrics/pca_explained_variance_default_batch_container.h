/* file: pca_explained_variance_default_batch_container.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the container for the multi-class confusion matrix.
//--
*/

#ifndef __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_CONTAINER_H__
#define __PCA_EXPLAINED_VARIANCE_DEFAULT_BATCH_CONTAINER_H__

#include "algorithms/pca/pca_explained_variance_batch.h"
#include "pca_explained_variance_default_batch_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace quality_metric
{
namespace explained_variance
{
using namespace daal::data_management;

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::ExplainedVarianceKernel, method, algorithmFPType);
}

template <typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    auto & explainedVariancesTable       = *(result->get(explainedVariances));
    auto & explainedVariancesRatiosTable = *(result->get(explainedVariancesRatios));
    auto & noiseVarianceTable            = *(result->get(noiseVariance));

    const auto & eigenvaluesTable = *(input->get(eigenvalues));

    daal::services::Environment::env & env = *_env;
    __DAAL_CALL_KERNEL(env, internal::ExplainedVarianceKernel, __DAAL_KERNEL_ARGUMENTS(method, algorithmFPType), compute, eigenvaluesTable,
                       explainedVariancesTable, explainedVariancesRatiosTable, noiseVarianceTable);
}

} // namespace explained_variance
} // namespace quality_metric
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
