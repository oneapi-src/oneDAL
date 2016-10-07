/* file: prelu_layer_forward.cpp */
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
//  Implementation of prelu calculation algorithm and types methods.
//--
*/

#include "prelu_layer_forward_types.h"
#include "prelu_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace forward
{
namespace interface1
{
/** \brief Default constructor */
Input::Input() {};

/**
* Returns dimensions of weights tensor
* \return Dimensions of weights tensor
*/
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *par) const
{
    const Parameter *parameter =  static_cast<const Parameter *>(par);
    services::SharedPtr<data_management::Tensor> dataTensor = get(layers::forward::data);

    size_t wStartDim = parameter->dataDimension;
    size_t wDimNumber = parameter->weightsDimension;

    services::Collection<size_t> _dims = dataTensor->getDimensions();
    services::Collection<size_t> _wdims;

    for (size_t i = wStartDim; i < wStartDim + wDimNumber; i++)
    {
        _wdims.push_back(_dims[i]);
    }
    return _wdims;
}

/**
 * Returns dimensions of biases tensor
 * \return Dimensions of biases tensor
 */
const services::Collection<size_t> Input::getBiasesSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}

/**
 * Checks input object of the forward prelu layer
 * \param[in] par     Layer parameter
 * \param[in] method  Computation method of the layer
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    layers::forward::Input::check(par, method);
    if(this->_errors->size() > 0) { return; }
    const Parameter *parameter =  static_cast<const Parameter *>(par);
    const services::Collection<size_t> &dataDimensions = get(layers::forward::data)->getDimensions();
    DAAL_CHECK_EX(parameter->dataDimension <= dataDimensions.size() - parameter->weightsDimension, services::ErrorIncorrectParameter,
                  services::ParameterName, dataDimensionStr());
    DAAL_CHECK_EX(parameter->weightsDimension != 0, services::ErrorIncorrectParameter, services::ParameterName, weightsDimensionStr());
    const services::Collection<size_t> weightsDimensions = getWeightsSizes(parameter);
    if (!data_management::checkTensor(get(layers::forward::weights).get(), this->_errors.get(), weightsStr(), &weightsDimensions)) { return; }
}

/** \brief Default constructor */
Result::Result() : layers::forward::Result() {};

/**
 * Returns the result of the forward prelu layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Result::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of the forward prelu layers
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    (*layerData)[id] = value;
}

/**
 * Checks the result of the forward prelu layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    layers::forward::Result::check(input, par, method);
    if(this->_errors->size() > 0) { return; }

    const Input *in = static_cast<const Input * >(input);
    const Parameter *parameter =  static_cast<const Parameter *>(par);
    const services::Collection<size_t> weightsDimension = in->getWeightsSizes(parameter);
    const services::Collection<size_t> &inputDimensions = in->get(layers::forward::data)->getDimensions();

    if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr(), &inputDimensions)) { return; }
    if(!parameter->predictionStage)
    {
        if (!data_management::checkTensor(get(prelu::auxData).get(), this->_errors.get(), auxDataStr(), &inputDimensions)) { return; }
        if (!data_management::checkTensor(get(prelu::auxWeights).get(), this->_errors.get(), auxWeightsStr(), &weightsDimension)) { return; }
    }
}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    return inputSize;
}

/**
 * Sets the result that is used in backward prelu layer
 * \param[in] input     Pointer to an object containing the input data
 */
void Result::setResultForBackward(const daal::algorithms::Input *input)
{
    const Input *in = static_cast<const Input *>(input);
    set(auxData, in->get(layers::forward::data));
    set(auxWeights, in->get(layers::forward::weights));
}

}// namespace interface1
}// namespace forward
}// namespace prelu
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
