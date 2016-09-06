/* file: lcn_layer_forward.cpp */
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
//  Implementation of lcn calculation algorithm and types methods.
//--
*/

#include "lcn_layer_forward_types.h"
#include "lcn_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lcn
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
    return services::Collection<size_t>();
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
* Checks input object of the forward local contrast normalization layer
* \param[in] parameter %Parameter of layer
* \param[in] method    Computation method of the layer
*/
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    layers::forward::Input::check(parameter, method);
    if( this->_errors->size() > 0 ) { return; }

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    services::SharedPtr<data_management::Tensor> dataTensor = get(layers::forward::data);
    if (!data_management::checkTensor(dataTensor.get(), this->_errors.get(), dataStr())) { return; }

    size_t nDims = dataTensor->getNumberOfDimensions();

    if( nDims != 4 )
    { this->_errors->add( services::ErrorIncorrectNumberOfDimensionsInTensor ); return; }
}
/**
 * Default constructor
 */
Result::Result() {}

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
 * Returns the result of forward local contrast normalization layer
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
 * Sets the result of forward local contrast normalization layer
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
    }
    (*layerData)[id] = value;
}

/**
 * Checks the result of the forward local contrast normalization layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    layers::forward::Result::check(input, par, method);
    if( this->_errors->size() > 0 ) { return; }

    services::SharedPtr<services::Error> error;

    const Input     *algInput     = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(par);

    services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
    if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

    const services::Collection<size_t> dataDims  = algInput->get(layers::forward::data)->getDimensions();;
    services::Collection<size_t> sigmaDims;
    getSigmaDimensions(algInput, algParameter, sigmaDims);

    services::Collection<size_t> cDims;
    getCDimensions(algInput, algParameter, cDims);

    if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr(), &dataDims)) { return; }
    if (!data_management::checkTensor(get(auxCenteredData).get(), this->_errors.get(), auxCenteredDataStr(), &dataDims)) { return; }
    if (!data_management::checkTensor(get(auxC).get(), this->_errors.get(), auxCStr(), &cDims)) { return; }
    if (!data_management::checkTensor(get(auxInvMax).get(), this->_errors.get(), auxInvMaxStr(), &sigmaDims)) { return; }
    if(!algParameter->predictionStage)
    {
        if (!data_management::checkTensor(get(auxSigma).get(), this->_errors.get(), auxSigmaStr(), &sigmaDims)) { return; }
    }
}

void Result::getSigmaDimensions(const Input *in, const Parameter *algParameter, services::Collection<size_t> &sigmaDims) const
{
    sigmaDims = in->get(layers::forward::data)->getDimensions();

    if(algParameter->sumDimension)
    {
        data_management::NumericTablePtr dimensionTable = algParameter->sumDimension;

        data_management::BlockDescriptor<int> block;
        dimensionTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *dataInt = block.getBlockPtr();
        size_t dim = dataInt[0];
        dimensionTable->releaseBlockOfRows(block);
        sigmaDims.erase(dim);
    }
}

void Result::getCDimensions(const Input *in, const Parameter *algParameter, services::Collection<size_t> &cDims) const
{
    getSigmaDimensions(in, algParameter, cDims);

    if(algParameter->sumDimension)
    {
        cDims.erase(algParameter->indices.dims[1] - 1);
        cDims.erase(algParameter->indices.dims[0] - 1);
    }
    else
    {
        cDims.erase(algParameter->indices.dims[1]);
        cDims.erase(algParameter->indices.dims[0]);
    }
}


}// namespace interface1
}// namespace forward
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
