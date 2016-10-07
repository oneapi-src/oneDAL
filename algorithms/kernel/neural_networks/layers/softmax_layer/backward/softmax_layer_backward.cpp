/* file: softmax_layer_backward.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of softmax calculation algorithm and types methods.
//--
*/

#include "softmax_layer_backward_types.h"
#include "softmax_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace softmax
{
namespace backward
{
namespace interface1
{
/** Default constructor */
Input::Input() {};

/**
* Returns an input object for the backward softmax layer
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
services::SharedPtr<data_management::Tensor> Input::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets an input object for the backward softmax layer
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    (*layerData)[id] = value;
}

/**
* Checks input object for the backward softmax layer
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const softmax::Parameter *parameter = static_cast<const softmax::Parameter *>(par);
    if (!parameter->propagateGradient) { return; }

    layers::backward::Input::check(par, method);
    if(this->_errors->size() > 0) { return; }

    const services::Collection<size_t>& inputDimensions = get(layers::backward::inputGradient)->getDimensions();

    if(!parameter->predictionStage)
    {
        if (!data_management::checkTensor(get(auxValue).get(), this->_errors.get(), auxValueStr(), &inputDimensions)) { return; }
    }

    DAAL_CHECK_EX(parameter->dimension < inputDimensions.size(), services::ErrorIncorrectParameter, services::ParameterName, dimensionStr());
}

Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward softmax layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const softmax::Parameter *parameter = static_cast<const softmax::Parameter *>(par);
    if (!parameter->propagateGradient) { return; }

    const Input *in = static_cast<const Input *>(input);
    if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(), &(in->get(auxValue)->getDimensions()))) { return; }
}

}// namespace interface1
}// namespace backward
}// namespace softmax
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
