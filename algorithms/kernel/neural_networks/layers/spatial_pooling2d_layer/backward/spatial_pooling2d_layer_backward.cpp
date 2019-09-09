/* file: spatial_pooling2d_layer_backward.cpp */
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
//  Implementation of spatial pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_pooling2d_layer_backward_types.h"
#include "spatial_pooling2d_layer_types.h"
#include "daal_strings.h"

using namespace daal::services;

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
Input::Input(const Input& other) : super(other) {}

/**
* Checks an input object for the backward 2D pooling layer
* \param[in] parameter Algorithm parameter
* \param[in] method Computation method
*/
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(parameter, method));

    size_t nDim = get(layers::backward::inputGradient)->getNumberOfDimensions();
    DAAL_CHECK_EX(nDim == 2, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ArgumentName, inputGradientStr());
    return s;
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
        return dims;

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
    size_t inputDim = (maskDim + 2 * padding - kernelSize + stride -1) / stride + 1;
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
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    const Input *algInput = static_cast<const Input *>(input);
    const services::Collection<size_t> &gradientDims = algInput->getGradientSize();

    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &gradientDims));

    size_t pyramidHeight = param->pyramidHeight;
    DAAL_CHECK_EX(pyramidHeight > 0, services::ErrorIncorrectParameter, services::ParameterName, pyramidHeightStr());
    services::Collection<size_t> valueDims = spatial_pooling2d::forward::Result::computeValueDimensions(get(layers::backward::gradient)->getDimensions(), param);
    DAAL_CHECK(valueDims[1] == algInput->get(layers::backward::inputGradient)->getDimensionSize(1), ErrorIncorrectParameter);

    size_t index0 = param->indices.size[0];
    size_t index1 = param->indices.size[1];
    unsigned int one = 1;
    size_t gradientNDims = 4;
    DAAL_CHECK_EX( index0 > 0 && index0 < gradientNDims && index1 > 0 && index1 < gradientNDims &&
                   index0 != index1, services::ErrorIncorrectParameter, services::ParameterName, indicesStr());
    if (gradientDims[index0] > gradientDims[index1]) { DAAL_CHECK_EX(one << (pyramidHeight - 1) <= gradientDims[index0], services::ErrorIncorrectParameter, services::ParameterName, pyramidHeightStr()); }
    if (gradientDims[index0] <= gradientDims[index1]) { DAAL_CHECK_EX(one << (pyramidHeight - 1) <= gradientDims[index1], services::ErrorIncorrectParameter, services::ParameterName, pyramidHeightStr()); }
    return s;
}

}// namespace interface1
}// namespace backward
}// namespace spatial_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
