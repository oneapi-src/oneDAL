/* file: layer_forward_container_base.cpp */
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
//  Implementation of neural_networks forward layer methods.
//--
*/

#include "layer_forward_container_base.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace forward
{
namespace interface1
{

services::Status LayerContainerIfaceImpl::completeInput()
{
    neural_networks::layers::forward::Input *input = static_cast<neural_networks::layers::forward::Input *>(_in);
    neural_networks::layers::forward::Result *result = static_cast<neural_networks::layers::forward::Result *>(_res);
    neural_networks::layers::Parameter *parameter = static_cast<neural_networks::layers::Parameter *>(_par);

    services::Status s;

    DAAL_CHECK_STATUS(s, allocateInput());
    DAAL_CHECK_STATUS(s, initializeInput());

    if(!parameter->predictionStage)
    {
        result->setResultForBackward(input);
    }
    return s;
}

services::Status LayerContainerIfaceImpl:: allocateInput()
{
    return services::Status();
}

/**
 * Initializes values of weights and biases if needed
 */
services::Status LayerContainerIfaceImpl::initializeInput()
{
    neural_networks::layers::Parameter *param = static_cast<neural_networks::layers::Parameter *>(_par);
    if( !param ) { return services::Status(); }
    const bool needToInitialize = !(param->weightsAndBiasesInitialized);

    services::Status s;
    services::SharedPtr<data_management::Tensor> tensor;

    neural_networks::layers::forward::Input *input = static_cast<neural_networks::layers::forward::Input *>(_in);

    tensor = input->get(weights);
    if (needToInitialize && tensor && tensor->getDimensions().size() )
    {
        param->weightsInitializer->input.set(initializers::data, tensor);
        s |= param->weightsInitializer->compute();
    }

    tensor = input->get(biases);
    if (needToInitialize && tensor && tensor->getDimensions().size() )
    {
        param->biasesInitializer->input.set(initializers::data, tensor);
        s |= param->biasesInitializer->compute();
    }

    param->weightsAndBiasesInitialized = true;
    return s;
}

}// namespace interface1
}// namespace forward
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
