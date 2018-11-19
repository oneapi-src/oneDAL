/* file: pooling2d_layer_backward.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
//  Implementation of pooling2d calculation algorithm and types methods.
//--
*/

#include "pooling2d_layer_backward_types.h"
#include "pooling2d_layer_types.h"
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
namespace pooling2d
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
    const data_management::NumericTablePtr inputDims = getAuxInputDimensions();
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

services::Collection<size_t> Input::getInputGradientSize(const pooling2d::Parameter *parameter) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    services::Collection<size_t> inputDims = getGradientSize();

    for (size_t d = 0; d < 2; d++)
    {
        inputDims[param->indices.size[d]] = computeInputDimension(
            inputDims[param->indices.size[d]], param->kernelSizes.size[d], param->paddings.size[d], param->strides.size[d]);
    }
    return inputDims;
}

size_t Input::computeInputDimension(size_t maskDim, size_t kernelSize, size_t padding, size_t stride) const
{
    size_t inputDim = (maskDim + 2 * padding - kernelSize + stride - 1) / stride + 1;
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

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Result::check(input, parameter, method));

    const Input *algInput = static_cast<const Input *>(input);

    //get expected gradient dimensions
    services::Collection<size_t> gradientDims = algInput->getGradientSize();
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &gradientDims));

    DAAL_CHECK_EX(param->strides.size[0] != 0 &&
                  param->strides.size[1] != 0, services::ErrorIncorrectParameter, services::ParameterName, stridesStr());

    size_t index0 = param->indices.size[0];
    size_t index1 = param->indices.size[1];
    DAAL_CHECK_EX(index0 <= gradientDims.size() - 1 && index1 <= gradientDims.size() - 1 &&
                  index0 != index1, services::ErrorIncorrectParameter, services::ParameterName, indicesStr());

    size_t kernelSize0 = param->kernelSizes.size[0];
    size_t kernelSize1 = param->kernelSizes.size[1];
    DAAL_CHECK_EX((kernelSize0 != 0 && kernelSize0 <= gradientDims[index0] + 2 * param->paddings.size[0] &&
                   kernelSize1 != 0 && kernelSize1 <= gradientDims[index1] + 2 * param->paddings.size[1]), services::ErrorIncorrectParameter, services::ParameterName,
                  kernelSizesStr());
    return s;
}

}// namespace interface1
}// namespace backward
}// namespace pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
