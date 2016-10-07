/* file: fullyconnected_layer_forward.cpp */
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

#include "fullyconnected_layer_forward_types.h"
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
namespace forward
{
namespace interface1
{
/**
 * Default constructor
 */
Input::Input() {};

/**
 * Returns dimensions of weights tensor
 * \return Dimensions of weights tensor
 */
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *parameter) const
{
    using daal::services::Collection;

    const Parameter *param =  static_cast<const Parameter *>(parameter);

    Collection<size_t> wDims = get(layers::forward::data)->getDimensions();
    wDims[0] = param->nOutputs;

    return wDims;
}

/**
 * Returns dimensions of biases tensor
 * \return Dimensions of biases tensor
 */
const services::Collection<size_t> Input::getBiasesSizes(const layers::Parameter *parameter) const
{
    using daal::services::Collection;

    const Parameter *param =  static_cast<const Parameter *>(parameter);
    size_t m = param->nOutputs;

    Collection<size_t> bDims;
    bDims.push_back( m );

    return bDims;
}

/**
 * Checks input object of the forward fully-connected layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    layers::forward::Input::check(parameter, method);
    if( this->_errors->size() > 0 ) { return; }

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    services::SharedPtr<data_management::Tensor> wTensor = get(layers::forward::weights);
    services::SharedPtr<data_management::Tensor> bTensor = get(layers::forward::biases);

    if( wTensor )
    {
    services::Collection<size_t> wDims = getWeightsSizes(algParameter);
    if (!data_management::checkTensor(wTensor.get(), this->_errors.get(), weightsStr(), &wDims)) { return; }
    }

    if( bTensor )
    {
    services::Collection<size_t> bDims = getBiasesSizes(algParameter);
    if (!data_management::checkTensor(bTensor.get(), this->_errors.get(), biasesStr(), &bDims)) { return; }
    }
}

/**
 * Default constructor
 */
Result::Result() {}

/**
 * Sets the result that is used in backward fully-connected layer
 * \param[in] input     Pointer to an object containing the input data
 */
void Result::setResultForBackward(const daal::algorithms::Input *input)
{
    const Input *in = static_cast<const Input *>(input);
    set(auxData,    in->get(layers::forward::data));
    set(auxWeights, in->get(layers::forward::weights));
}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    const Parameter *param =  static_cast<const Parameter *>(par);

    services::Collection<size_t> valueDims;
    valueDims.push_back(inputSize[0]);
    valueDims.push_back(param->nOutputs);
    return valueDims;
}

/**
 * Returns the result of forward fully-connected layer
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Result::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(!layerData)
    {
        this->_errors->add(services::ErrorNullLayerData);
        return services::SharedPtr<data_management::Tensor>();
    }
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of forward fully-connected layer
 * \param[in] id     Identifier of the result
 * \param[in] value  Result
 */
void Result::set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(!layerData)
    {
        this->_errors->add(services::ErrorNullLayerData);
        return;
    }
    (*layerData)[id] = value;
}

/**
 * Checks the result of the forward fully-connected layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    layers::forward::Result::check(input, par, method);
    if( this->_errors->size() > 0 ) { return; }

    const Input     *algInput     = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(par);

    services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
    if (!layerData && algParameter->predictionStage == false) { this->_errors->add(services::ErrorNullLayerData); return; }

    services::SharedPtr<data_management::Tensor> dataTensor  = algInput->get(layers::forward::data);

    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
    const services::Collection<size_t>     wDims = algInput->getWeightsSizes(algParameter);
    const services::Collection<size_t>   valDims = getValueSize(dataDims, algParameter, defaultDense);

    if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr(), &valDims)) { return; }
    if(!algParameter->predictionStage)
    {
        if (!data_management::checkTensor(get(auxData).get(), this->_errors.get(), auxDataStr(), &dataDims)) { return; }
        if (algParameter->propagateGradient)
        {
            if (!data_management::checkTensor(get(auxWeights).get(), this->_errors.get(), auxWeightsStr(), &wDims)) { return; }
        }
    }
}

}// namespace interface1
}// namespace forward
}// namespace fullyconnected
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
