/* file: batch_normalization_layer_backward.cpp */
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

#include "batch_normalization_layer_backward_types.h"
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
namespace backward
{
namespace interface1
{
/** Default constructor */
Input::Input() {};

/**
 * Returns an input object for the backward batch normalization layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Input::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*inputData)[id]);
}

/**
 * Sets an input object for the backward batch normalization layer
 * \param[in] id  Identifier of the input object
 * \param[in] ptr Pointer to the object
 */
void Input::set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &ptr)
{
    services::SharedPtr<layers::LayerData> inputData = get(layers::backward::inputFromForward);
    (*inputData)[id] = ptr;
}

/**
 * Checks an input object for the backward batch normalization layer
 * \param[in] parameter Layer parameter
 * \param[in] method    Computation method
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    layers::backward::Input::check(parameter, method);
    if( this->_errors->size() > 0 ) { return; }

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    services::SharedPtr<data_management::Tensor> inputGradientTensor = get(layers::backward::inputGradient);
    if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr())) { return; }
    const services::Collection<size_t> &dataDims = inputGradientTensor->getDimensions();

    size_t dimension = algParameter->dimension;
    double epsilon = algParameter->epsilon;
    DAAL_CHECK_EX(dimension <= dataDims.size(), services::ErrorIncorrectParameter, services::ParameterName, dimensionStr());
    DAAL_CHECK_EX(epsilon > 0.0 && epsilon < 1.0, services::ErrorIncorrectParameter, services::ParameterName, epsilonStr());

    size_t dimensionSize = dataDims[dimension];
    services::Collection<size_t> auxDims(1);
    auxDims[0] = dimensionSize;

    if (!data_management::checkTensor(get(auxData).get(),               this->_errors.get(), auxDataStr(),              &dataDims)) { return; }
    if (!data_management::checkTensor(get(auxWeights).get(),            this->_errors.get(), auxWeightsStr(),            &auxDims)) { return; }
    if (!data_management::checkTensor(get(auxMean).get(),               this->_errors.get(), auxMeanStr(),               &auxDims)) { return; }
    if (!data_management::checkTensor(get(auxStandardDeviation).get(),  this->_errors.get(), auxStandardDeviationStr(),  &auxDims)) { return; }
    if (!data_management::checkTensor(get(auxPopulationMean).get(),     this->_errors.get(), auxPopulationMeanStr(),     &auxDims)) { return; }
    if (!data_management::checkTensor(get(auxPopulationVariance).get(), this->_errors.get(), auxPopulationVarianceStr(), &auxDims)) { return; }
}

/** Default constructor */
Result::Result() {}

/**
 * Checks the result of the backward batch normalization layer
 * \param[in] input     %Input object for the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t dimension = algParameter->dimension;

    services::SharedPtr<data_management::Tensor> inputGradientTensor = algInput->get(layers::backward::inputGradient);
    const services::Collection<size_t> &gradientDims = inputGradientTensor->getDimensions();

    if (algParameter->propagateGradient)
    {
        if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(), &gradientDims)) { return; }
    }

    size_t dimensionSize = gradientDims[dimension];
    services::Collection<size_t> derDims(1);
    derDims[0] = dimensionSize;

    if (!data_management::checkTensor(get(layers::backward::weightDerivatives).get(), this->_errors.get(), weightDerivativesStr(), &derDims)) { return; }
    if (!data_management::checkTensor(get(layers::backward::biasDerivatives).get(), this->_errors.get(), biasDerivativesStr(), &derDims)) { return; }
}

}// namespace interface1
}// namespace backward
}// namespace batch_normalization
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
