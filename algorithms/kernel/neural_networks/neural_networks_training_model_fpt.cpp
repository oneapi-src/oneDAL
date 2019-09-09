/* file: neural_networks_training_model_fpt.cpp */
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
//  Implementation of model of the training stage of neural network
//--
*/

#include "neural_networks_training_model.h"
#include "neural_networks_weights_and_biases.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{

template class TensorWeightsAndBiasesDerivatives<DAAL_FPTYPE>;
template class NumericTableWeightsAndBiasesDerivatives<DAAL_FPTYPE>;

namespace training
{

template<typename modelFPType>
services::Status DAAL_EXPORT Model::createWeightsAndBiasesDerivatives()
{
    using namespace services;
    Status s;
    if (_storeWeightDerivativesInTable)
    {
        _weightsAndBiasesDerivatives = NumericTableWeightsAndBiasesDerivatives<modelFPType>::create(
                _forwardLayers, _backwardLayers, &s);
    }
    else
    {
        _weightsAndBiasesDerivatives = TensorWeightsAndBiasesDerivatives<modelFPType>::create(
                _backwardLayers, &s);
    }
    return s;
}

template DAAL_EXPORT services::Status Model::createWeightsAndBiasesDerivatives<DAAL_FPTYPE>();

}
}
}
}
