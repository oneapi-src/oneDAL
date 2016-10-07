/* file: spatial_pooling2d_layer_backward.cpp */
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
//  Implementation of spatial pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_pooling2d_layer_backward_types.h"
#include "spatial_pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_pooling2d
{
namespace backward
{
namespace interface1
{
/** Default constructor */
Input::Input() {}

/**
* Checks an input object for the backward 2D pooling layer
* \param[in] parameter Algorithm parameter
* \param[in] method Computation method
*/
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return; }

    data_management::TensorPtr inputGradientTensor = get(layers::backward::inputGradient);
    if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr())) { return; }

    DAAL_CHECK_EX(get(layers::backward::inputGradient)->getDimensions().size() == 2, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ArgumentName, inputGradientStr());

    size_t nDim = inputGradientTensor->getNumberOfDimensions();
    DAAL_CHECK(nDim == 2, ErrorIncorrectParameter);
}

/**
 * Return the collection with gradient size
 * \return The collection with gradient size
 */
services::Collection<size_t> Input::getGradientSize() const
{
    services::Collection<size_t> dims;
    const data_management::NumericTablePtr inputDims = getAuxInputDimensions();
    if (!inputDims)
    { this->_errors->add(services::ErrorNullInputNumericTable); return dims; }

    data_management::BlockDescriptor<int> block;
    inputDims->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *inputDimsArray = block.getBlockPtr();
    for(size_t i = 0; i < inputDims->getNumberOfColumns(); i++)
    {
        dims.push_back((size_t) inputDimsArray[i]);
    }
    inputDims->releaseBlockOfRows(block);
    return dims;
}

size_t Input::computeInputDimension(size_t maskDim, size_t kernelSize, size_t padding, size_t stride) const
{
    size_t inputDim = (maskDim + 2 * padding - kernelSize + stride) / stride;
    return inputDim;
}

    /** Default constructor */
Result::Result() {}

/**
* Checks the result of the backward 2D pooling layer
* \param[in] input %Input object for the layer
* \param[in] parameter %Parameter of the layer
* \param[in] method Computation method
*/
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return; }

    const Input *algInput = static_cast<const Input *>(input);
    const services::Collection<size_t> &gradientDims = algInput->getGradientSize();

    if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(), &gradientDims)) { return; }

    services::Collection<size_t> valueDims = spatial_pooling2d::forward::Result::computeValueDimensions(get(layers::backward::gradient)->getDimensions(), param);
    DAAL_CHECK(valueDims[1] == algInput->get(layers::backward::inputGradient)->getDimensionSize(1), ErrorIncorrectParameter);
}

}// namespace interface1
}// namespace backward
}// namespace spatial_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
