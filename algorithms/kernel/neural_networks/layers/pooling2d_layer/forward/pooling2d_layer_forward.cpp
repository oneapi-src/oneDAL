/* file: pooling2d_layer_forward.cpp */
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

#include "pooling2d_layer_forward_types.h"
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
namespace forward
{
namespace interface1
{
/**
 * Default constructor
 */
Input::Input() : layers::forward::Input() {}
Input::Input(const Input& other) : super(other) {}

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
 * Checks an input object for the 2D pooling layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Input::check(parameter, method));

    const Parameter *param = static_cast<const Parameter *>(parameter);
    const services::Collection<size_t> &dataDims = get(layers::forward::data)->getDimensions();

    DAAL_CHECK_EX(dataDims.size() >= 2, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ParameterName, dataStr());

    DAAL_CHECK_EX(param->strides.size[0] != 0 &&
                  param->strides.size[1] != 0, services::ErrorIncorrectParameter, services::ParameterName, stridesStr());

    size_t index0 = param->indices.size[0];
    size_t index1 = param->indices.size[1];
    DAAL_CHECK_EX(index0 <= dataDims.size() - 1 && index1 <= dataDims.size() - 1 &&
                  index0 != index1, services::ErrorIncorrectParameter, services::ParameterName, indicesStr());

    size_t kernelSize0 = param->kernelSizes.size[0];
    size_t kernelSize1 = param->kernelSizes.size[1];
    DAAL_CHECK_EX((kernelSize0 != 0 && kernelSize0 <= dataDims[index0] + 2 * param->paddings.size[0] &&
                   kernelSize1 != 0 && kernelSize1 <= dataDims[index1] + 2 * param->paddings.size[1]), services::ErrorIncorrectParameter, services::ParameterName,
                  kernelSizesStr());
    return s;
}

/** Default constructor */
Result::Result() {}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    services::Collection<size_t> valueDims(inputSize);
    computeValueDimensions(valueDims, static_cast<const Parameter *>(par));
    return valueDims;
}

/**
 * Checks the result of the forward 2D pooling layer
 * \param[in] input %Input object for the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, parameter, method));

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr dataTensor = algInput->get(layers::forward::data);
    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
    services::Collection<size_t> valueDims(dataDims);

    computeValueDimensions(valueDims, param);

    return data_management::checkTensor(get(layers::forward::value).get(), valueStr(), &valueDims);
}

size_t Result::computeValueDimension(size_t dataDim, size_t kernelSize, size_t padding, size_t stride) const
{
    size_t valueDim = (dataDim + 2 * padding - kernelSize + stride - 1) / stride + 1;
    return valueDim;
}

void Result::computeValueDimensions(services::Collection<size_t> &dims, const Parameter *param) const
{
    for (size_t d = 0; d < 2; d++)
    {
        dims[param->indices.size[d]] =
            computeValueDimension(dims[param->indices.size[d]], param->kernelSizes.size[d], param->paddings.size[d], param->strides.size[d]);
    }
}

data_management::NumericTablePtr Result::createAuxInputDimensions(const services::Collection<size_t> &dataDims) const
{
    size_t nInputDims = dataDims.size();
    services::SharedPtr<data_management::HomogenNumericTable<int> > auxInputDimsTable = data_management::HomogenNumericTable<int>::create(
        nInputDims, 1, data_management::NumericTableIface::doAllocate);

    if (!auxInputDimsTable) { return data_management::NumericTablePtr(); }
    if (auxInputDimsTable->getArray() == 0) { return data_management::NumericTablePtr(); }

    int *auxInputDimsData = auxInputDimsTable->getArray();
    for (size_t i = 0; i < nInputDims; i++)
    {
        auxInputDimsData[i] = (int)dataDims[i];
    }
    return auxInputDimsTable;
}

}// namespace interface1
}// namespace forward
}// namespace pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
