/* file: spatial_pooling2d_layer_forward.cpp */
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
//  Implementation of spatial pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_pooling2d_layer_types.h"
#include "spatial_pooling2d_layer_forward_types.h"
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
namespace forward
{
namespace interface1
{
/**
 * Default constructor
 */
Input::Input() {}
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
    data_management::TensorPtr dataTensor = get(layers::forward::data);
    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
    size_t nDim = dataDims.size();
    DAAL_CHECK_EX(nDim >= 2, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ParameterName, dataStr());

    size_t index0 = param->indices.size[0];
    size_t index1 = param->indices.size[1];
    size_t pyramidHeight = param->pyramidHeight;
    unsigned int one = 1;
    DAAL_CHECK_EX(0 < index0 && index0 < nDim && 0 < index1 && index1 < nDim &&
                  index0 != index1, services::ErrorIncorrectParameter, services::ParameterName, indicesStr());

    DAAL_CHECK_EX(pyramidHeight > 0, services::ErrorIncorrectParameter, services::ParameterName, pyramidHeightStr());
    if (dataDims[index0] > dataDims[index1])
    {
        DAAL_CHECK_EX(one << (pyramidHeight - 1) <= dataDims[index0], services::ErrorIncorrectParameter, services::ParameterName, pyramidHeightStr());
    }
    if (dataDims[index0] <= dataDims[index1])
    {
        DAAL_CHECK_EX(one << (pyramidHeight - 1) <= dataDims[index1], services::ErrorIncorrectParameter, services::ParameterName, pyramidHeightStr());
    }
    DAAL_CHECK_EX(dataTensor->getDimensions().size() == 4, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ArgumentName, dataStr());
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
    services::Collection<size_t> valueDims = computeValueDimensions(inputSize, static_cast<const Parameter *>(par));
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
    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(parameter);

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, parameter, method));

    data_management::TensorPtr dataTensor = algInput->get(layers::forward::data);
    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();

    DAAL_CHECK_EX(get(layers::forward::value)->getDimensions().size() == 2, services::ErrorIncorrectNumberOfDimensionsInTensor, services::ArgumentName, valueStr());

    services::Collection<size_t> valueDims = getValueSize(dataDims, param, method);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::forward::value).get(), valueStr(), &valueDims));
    if(!param->predictionStage)
    {
        LayerDataPtr layerData = get(layers::forward::resultForBackward);

        if (layerData->size() != 1)
            if (!layerData) { return services::Status(services::ErrorIncorrectSizeOfLayerData); }
    }
    return s;
}

services::Collection<size_t> Result::computeValueDimensions(const services::Collection<size_t> &inputDims, const Parameter *param)
{
    services::Collection<size_t> valueDims(2);
    valueDims[0] = inputDims[0];
    size_t pow4 = 1;
    for(size_t i = 0; i < param->pyramidHeight; i++)
    {
        pow4 *= 4;
    }
    valueDims[1] = inputDims[6 - param->indices.size[0] - param->indices.size[1]] * (pow4 - 1) / 3;
    return valueDims;
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
}// namespace spatial_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
