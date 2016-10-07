/* file: fullyconnected_layer_backward.cpp */
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
//  Implementation of fullyconnected calculation algorithm and types methods.
//--
*/

#include "fullyconnected_layer_backward_types.h"
#include "fullyconnected_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
namespace backward
{
namespace interface1
{
/**
 * Default constructor
 */
Input::Input() {};

/**
 * Returns an input object for backward fully-connected layer
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
 * Sets input for the backward fully-connected layer
 * \param[in] id    Identifier of the input  object
 * \param[in] value Input object to set
 */
void Input::set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the fully-connected layer
 * \param[in] par       %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    layers::backward::Input::check(par, method);
    if( this->_errors->size() > 0 ) { return; }

    const Parameter *param = static_cast<const Parameter*>(par);

    services::SharedPtr<data_management::Tensor> xTensor = get(auxData);

    if (!data_management::checkTensor(xTensor.get(), this->_errors.get(), auxDataStr())) { return; }

    const services::Collection<size_t> &xDims = xTensor->getDimensions();
    const services::Collection<size_t> &gDims = get(layers::backward::inputGradient)->getDimensions();

    services::Collection<size_t> gradDims;
    gradDims.push_back(xDims[0]);
    gradDims.push_back(param->nOutputs);

    if (!data_management::checkTensor(get(layers::backward::inputGradient).get(), this->_errors.get(), inputGradientStr(), &gradDims)) { return; }
    if (param->propagateGradient)
    {
        services::Collection<size_t> wDims = xDims;
        wDims[0] = param->nOutputs;
        if (!data_management::checkTensor(get(auxWeights).get(), this->_errors.get(), auxWeightsStr(), &wDims)) { return; }
    }
}

/**
 * Default constructor
 */
Result::Result() : layers::backward::Result() {}

/**
 * Checks the result of the fully-connected layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    layers::backward::Result::check(input, par, method);
    if( this->_errors->size() > 0 ) { return; }

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(par);

    if (param->propagateGradient)
    {
        if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(),
                                          &(algInput->get(auxData)->getDimensions()))) { return; }
    }
    if (!data_management::checkTensor(get(layers::backward::weightDerivatives).get(), this->_errors.get(), weightDerivativesStr(),
                                      &(algInput->get(auxWeights)->getDimensions()))) { return; }

    services::Collection<size_t> bDims;
    bDims.push_back(param->nOutputs);

    if (!data_management::checkTensor(get(layers::backward::biasDerivatives).get(), this->_errors.get(), biasDerivativesStr(), &bDims)) { return; }
}

}// namespace interface1
}// namespace backward
}// namespace fullyconnected
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
