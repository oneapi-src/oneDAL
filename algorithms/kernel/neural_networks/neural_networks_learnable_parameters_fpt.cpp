/* file: neural_networks_learnable_parameters_fpt.cpp */
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
//  Implementation of classes for storing learnable parameters of neural network
//--
*/

#include "neural_networks_learnable_parameters.h"
#include "neural_networks_weights_and_biases.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{

template class TensorWeightsAndBiases<DAAL_FPTYPE>;
template class NumericTableWeightsAndBiases<DAAL_FPTYPE>;

template<typename modelFPType>
services::Status DAAL_EXPORT ModelImpl::createWeightsAndBiases(bool checkAllocation)
{
    using namespace services;
    if (_weightsAndBiasesCreated) { return services::Status(); }
    services::Status s;

    if (checkAllocation)
    {
        DAAL_CHECK_STATUS(s, checkWeightsAndBiasesAllocation());
    }

    if (_storeWeightsInTable)
    {
        _weightsAndBiases = NumericTableWeightsAndBiases<modelFPType>::create(_forwardLayers, &s);
    }
    else
    {
        _weightsAndBiases = TensorWeightsAndBiases<modelFPType>::create(_forwardLayers, &s);
    }
    if (s)
        _weightsAndBiasesCreated = true;

    return s;
}

template DAAL_EXPORT services::Status ModelImpl::createWeightsAndBiases<DAAL_FPTYPE>(bool checkAllocation);

}
}
}
