/* file: batch_normalization_layer_forward_batch_container.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of forward batch normalization layer container.
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_BATCH_CONTAINER_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_BATCH_CONTAINER_H__

#include "neural_networks/layers/batch_normalization/batch_normalization_layer_forward.h"
#include "batch_normalization_layer_forward_kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace forward
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::BatchNormalizationKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    batch_normalization::forward::Input *input = static_cast<batch_normalization::forward::Input *>(_in);
    batch_normalization::forward::Result *result = static_cast<batch_normalization::forward::Result *>(_res);

    batch_normalization::Parameter *parameter = static_cast<batch_normalization::Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    Tensor *inputTensor                   = input->get(layers::forward::data).get();
    Tensor *inputPopulationMeanTensor     = input->get(populationMean).get();
    Tensor *inputPopulationVarianceTensor = input->get(populationVariance).get();
    Tensor *weightsTensor                 = input->get(layers::forward::weights).get();
    Tensor *biasesTensor                  = input->get(layers::forward::biases).get();
    Tensor *populationMeanTensor          = result->get(auxPopulationMean).get();
    Tensor *populationVarianceTensor      = result->get(auxPopulationVariance).get();
    Tensor *valueTensor                   = result->get(layers::forward::value).get();
    Tensor *meanTensor                    = result->get(auxMean).get();
    Tensor *stDevTensor                   = result->get(auxStandardDeviation).get();

    __DAAL_CALL_KERNEL(env, internal::BatchNormalizationKernel, __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, inputTensor, inputPopulationMeanTensor,
        inputPopulationVarianceTensor, weightsTensor, biasesTensor, populationMeanTensor, populationVarianceTensor, valueTensor, meanTensor,
        stDevTensor, parameter);
}
} // namespace interface1
} // namespace forward
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
