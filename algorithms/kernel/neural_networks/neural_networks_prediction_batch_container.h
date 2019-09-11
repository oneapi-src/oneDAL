/* file: neural_networks_prediction_batch_container.h */
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
//  Implementation of neural_networks calculation algorithm container.
//--
*/

#ifndef __NEURAL_NETWORKS_PREDICTION_BATCH_CONTAINER_H__
#define __NEURAL_NETWORKS_PREDICTION_BATCH_CONTAINER_H__

#include "neural_networks/neural_networks_prediction.h"
#include "neural_networks_types.h"
#include "neural_networks_prediction_types.h"
#include "neural_networks_prediction_feedforward_kernel.h"
#include "kernel.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace prediction
{
namespace interface1
{
template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::BatchContainer(daal::services::interface1::Environment::env *daalEnv)
{
    __DAAL_INITIALIZE_KERNELS(internal::NeuralNetworksFeedforwardPredictionKernel, algorithmFPType, method);
}

template<typename algorithmFPType, Method method, CpuType cpu>
BatchContainer<algorithmFPType, method, cpu>::~BatchContainer()
{
    __DAAL_DEINITIALIZE_KERNELS();
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::compute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NeuralNetworksFeedforwardPredictionKernel,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), compute, input, result);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::setupCompute()
{
    Input *input = static_cast<Input *>(_in);
    Result *result = static_cast<Result *>(_res);

    Parameter *parameter = static_cast<Parameter *>(_par);
    daal::services::Environment::env &env = *_env;

    __DAAL_CALL_KERNEL(env, internal::NeuralNetworksFeedforwardPredictionKernel,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), initialize, input, parameter, result);
}

template<typename algorithmFPType, Method method, CpuType cpu>
services::Status BatchContainer<algorithmFPType, method, cpu>::resetCompute()
{
    daal::services::Environment::env &env = *_env;
    __DAAL_CALL_KERNEL(env, internal::NeuralNetworksFeedforwardPredictionKernel,
        __DAAL_KERNEL_ARGUMENTS(algorithmFPType, method), reset);
}
} // namespace interface1
}
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
