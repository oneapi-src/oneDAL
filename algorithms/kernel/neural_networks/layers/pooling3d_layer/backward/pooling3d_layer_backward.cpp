/* file: pooling3d_layer_backward.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of pooling3d calculation algorithm and types methods.
//--
*/

#include "pooling3d_layer_backward_types.h"
#include "pooling3d_layer_types.h"
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
namespace pooling3d
{
namespace backward
{
namespace interface1
{
/** Default constructor */
Input::Input() {}
Input::Input(const Input& other) : super(other) {}

/**
 * Return the collection with gradient size
 * \return The collection with gradient size
 */
services::Collection<size_t> Input::getGradientSize() const
{
    services::Collection<size_t> dims;
    data_management::NumericTablePtr inputDims = getAuxInputDimensions();
    if(!data_management::checkNumericTable(inputDims.get(), auxInputDimensionsStr())) { return dims; }

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

services::Collection<size_t> Input::getInputGradientSize(const pooling3d::Parameter *parameter) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    services::Collection<size_t> inputDims = getGradientSize();

    for (size_t d = 0; d < 3; d++)
    {
        inputDims[param->indices.size[d]] = computeInputDimension(
            inputDims[param->indices.size[d]], param->kernelSizes.size[d], param->paddings.size[d], param->strides.size[d]);
    }
    return inputDims;
}

size_t Input::computeInputDimension(size_t maskDim, size_t kernelSize, size_t padding, size_t stride) const
{
    size_t inputDim = (maskDim + 2 * padding - kernelSize + stride) / stride;
    return inputDim;
}

/** Default constructor */
Result::Result() {}

/**
 * Checks the result of the backward 3D pooling layer
 * \param[in] input %Input object for the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Result::check(input, parameter, method));

    const Input *algInput = static_cast<const Input *>(input);

    //get expected gradient dimensions
    const services::Collection<size_t> &gradientDims = algInput->getGradientSize();
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &gradientDims));

    for (size_t i = 0; i < 3; i++)
    {
        size_t index = param->indices.size[i];
        size_t stride = param->strides.size[i];
        size_t kernelSize = param->kernelSizes.size[i];
        size_t padding = param->paddings.size[i];

        DAAL_CHECK_EX(stride != 0, services::ErrorIncorrectParameter, services::ParameterName, stridesStr());
        DAAL_CHECK_EX(index <= gradientDims.size() - 1, services::ErrorIncorrectParameter, services::ParameterName,
                      indicesStr());
        DAAL_CHECK_EX((kernelSize != 0 &&
                       kernelSize <= gradientDims[index] + 2 * padding), services::ErrorIncorrectParameter, services::ParameterName, kernelSizesStr());
    }

    DAAL_CHECK_EX(param->indices.size[0] != param->indices.size[1] &&
                  param->indices.size[1] != param->indices.size[2], services::ErrorIncorrectParameter, services::ParameterName,
                  indicesStr());
    return s;
}

}// namespace interface1
}// namespace backward
}// namespace pooling3d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
