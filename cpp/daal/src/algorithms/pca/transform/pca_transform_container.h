/* file: pca_transform_container.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of pca transformation algorithm container -- a class
//  that contains fast pca transformation kernels
//  for supported architectures.
//--
*/

#ifndef __PCA_TRANSFORM_CONTAINER_H__
#define __PCA_TRANSFORM_CONTAINER_H__

#include "src/algorithms/pca/transform/pca_transform_kernel.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace transform
{
template <typename algorithmFPType, transform::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env * daalEnv) : AnalysisContainerIface<batch>(daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::TransformKernel, algorithmFPType, method);
}

template <typename algorithmFPType, transform::Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template <typename algorithmFPType, transform::Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input * input   = static_cast<Input *>(_in);
    Result * result = static_cast<Result *>(_res);

    bool hasTransform           = input->get(dataForTransform).get() != nullptr;
    NumericTable * pMeans       = hasTransform ? input->get(dataForTransform, mean).get() : NULL;
    NumericTable * pVariances   = hasTransform ? input->get(dataForTransform, variance).get() : NULL;
    NumericTable * pEigenvalues = hasTransform ? input->get(dataForTransform, eigenvalue).get() : NULL;

    daal::services::Environment::env & env = *_env;

    __DAAL_CALL_KERNEL(env, internal::TransformKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, *(input->get(data)),
                       *(input->get(eigenvectors)), pMeans, pVariances, pEigenvalues, *(result->get(transformedData)));
}

} // namespace transform
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
