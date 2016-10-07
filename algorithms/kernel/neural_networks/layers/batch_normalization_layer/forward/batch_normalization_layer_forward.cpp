/* file: batch_normalization_layer_forward.cpp */
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
//  Implementation of batch normalization calculation algorithm and types methods.
//--
*/

#include "batch_normalization_layer_forward_types.h"
#include "batch_normalization_layer_types.h"

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
/** Default constructor */
Input::Input() : layers::forward::Input(5) {}

/**
 * Returns dimensions of weights tensor
 * \return Dimensions of weights tensor
 */
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *parameter) const
{
    const Parameter *algParameter =  static_cast<const Parameter *>(parameter);
    const services::Collection<size_t> &dims = get(layers::forward::data)->getDimensions();
    services::Collection<size_t> wDims(1);
    wDims[0] = dims[algParameter->dimension];
    return wDims;
}

/**
 * Returns dimensions of biases tensor
 * \return Dimensions of biases tensor
 */
const services::Collection<size_t> Input::getBiasesSizes(const layers::Parameter *parameter) const
{
    return getWeightsSizes(parameter);
}

/**
 * Returns an input object for forward batch normalization layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Input::get(InputLayerDataId id) const
{
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the forward batch normalization layer
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Input object to set
 */
void Input::set(InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks input object of the forward batch normalization layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    services::SharedPtr<data_management::Tensor> dataTensor = get(layers::forward::data);
    if (!data_management::checkTensor(dataTensor.get(), this->_errors.get(), dataStr())) { return; }

    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();

    size_t dimension = algParameter->dimension;
    double alpha = algParameter->alpha;
    double epsilon = algParameter->epsilon;
    DAAL_CHECK_EX(dimension <= dataDims.size(), services::ErrorIncorrectParameter, services::ParameterName, dimensionStr());
    DAAL_CHECK_EX((alpha > 0.0 && alpha < 1.0), services::ErrorIncorrectParameter, services::ParameterName, alphaStr());
    DAAL_CHECK_EX(epsilon > 0.0 && epsilon < 1.0, services::ErrorIncorrectParameter, services::ParameterName, epsilonStr());

    size_t dimensionSize = dataTensor->getDimensionSize(dimension);
    services::Collection<size_t> weightDims(1);
    weightDims[0] = dimensionSize;
    if (!data_management::checkTensor(get(layers::forward::weights).get(), this->_errors.get(), weightsStr(), &weightDims)) { return; }
    if (!data_management::checkTensor(get(layers::forward::biases).get(),  this->_errors.get(), biasesStr(), &weightDims)) { return; }
    if (!data_management::checkTensor(get(populationMean).get(),           this->_errors.get(), populationMeanStr(), &weightDims)) { return; }
    if (!data_management::checkTensor(get(populationVariance).get(),       this->_errors.get(), populationVarianceStr(), &weightDims)) { return; }

}

/** Default Constructor */
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
 * Sets the result that is used in backward batch normalization layer
 * \param[in] input     Pointer to an object containing the input data
 */
void Result::setResultForBackward(const daal::algorithms::Input *input)
{
    const Input *in = static_cast<const Input *>(input);
    set(auxData,    in->get(layers::forward::data));
    set(auxWeights, in->get(layers::forward::weights));
}

/**
 * Returns the result of the forward batch normalization layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Result::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of the forward batch normalization layer
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Result
 */
void Result::set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &ptr)
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
    (*layerData)[id] = ptr;
}

/**
 * Checks the result of the forward batch normalization layer
 * \param[in] input     %Input of the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method of the layer
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t dimension = algParameter->dimension;

    services::SharedPtr<data_management::Tensor> dataTensor = algInput->get(layers::forward::data);
    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();

    services::SharedPtr<data_management::Tensor> valueTensor = get(layers::forward::value);
    if (!data_management::checkTensor(valueTensor.get(), this->_errors.get(), valueStr(), &dataDims)) { return; }

    size_t dimensionSize = valueTensor->getDimensionSize(dimension);
    services::Collection<size_t> auxDims(1);
    auxDims[0] = dimensionSize;

    services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
    if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

    if (algParameter->predictionStage == false && layerData->size() != 6)
        if (!layerData) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }

    if (algParameter->predictionStage == true && layerData->size() != 2)
        if (!layerData) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }

    if (!data_management::checkTensor(get(auxMean).get(),               this->_errors.get(), auxMeanStr(), &auxDims)) { return; }
    if (!data_management::checkTensor(get(auxStandardDeviation).get(),  this->_errors.get(), auxStandardDeviationStr(), &auxDims)) { return; }
    if(!algParameter->predictionStage)
    {
        if (!data_management::checkTensor(get(auxData).get(),               this->_errors.get(), auxDataStr(), &dataDims)) { return; }
        if (!data_management::checkTensor(get(auxWeights).get(),            this->_errors.get(), auxWeightsStr(), &auxDims)) { return; }
        if (!data_management::checkTensor(get(auxPopulationMean).get(),     this->_errors.get(), auxPopulationMeanStr(), &auxDims)) { return; }
        if (!data_management::checkTensor(get(auxPopulationVariance).get(), this->_errors.get(), auxPopulationVarianceStr(), &auxDims)) { return; }
    }
}
}// namespace interface1
}// namespace forward
}// namespace batch_normalization
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
