/* file: reshape_layer_forward.cpp */
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
//  Implementation of reshape calculation algorithm and types methods.
//--
*/

#include "reshape_layer_forward_types.h"
#include "reshape_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace reshape
{
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_RESHAPE_FORWARD_RESULT_ID);
/** Default constructor */
Input::Input() {};
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

/** Default constructor */
Result::Result() : layers::forward::Result() {};

/**
 * Returns result of the forward reshape layer
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(!layerData)
        return data_management::NumericTablePtr();
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of the forward reshape layer
 * \param[in] id      Identifier of the result
 * \param[in] value   Pointer to the object
 */
void Result::set(LayerDataId id, const data_management::NumericTablePtr &value)
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(layerData)
        (*layerData)[id] = value;
}

/**
 * Checks the result of the forward reshape layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, parameter, method));

    const Input *algInput = static_cast<const Input *>(input);
    const layers::reshape::Parameter *par = static_cast<const layers::reshape::Parameter * >(parameter);
    const services::Collection<size_t>& inDims = algInput->get(layers::forward::data)->getDimensions();
    services::Collection<size_t> outDims = par->reshapeDimensions;

    bool haveNegative = false;
    size_t negIndex = 0;
    size_t nonNegSize = 1;

    for( size_t i = 0; i < outDims.size(); i++ )
    {
        if( outDims[i] == undefinedDimensionSize )
        {
            haveNegative = true;
            negIndex = i;
        }
        else
        {
            if( outDims[i] == 0 )
            {
                outDims[i] = inDims[i];
            }

            nonNegSize *= outDims[i];
        }
    }

    if(haveNegative)
    {
        outDims[negIndex] = algInput->get(layers::forward::data)->getSize() / nonNegSize;
    }

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::forward::value).get(), valueStr(), &outDims));

    if(!par->predictionStage)
    {
        data_management::NumericTablePtr dimsNT = get(auxInputDimensions);
        if (dimsNT.get() == 0) { return services::Status(services::ErrorNullNumericTable); }
        if (dimsNT->getNumberOfColumns() != inDims.size()) { return services::Status(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); }
        if (dimsNT->getNumberOfRows() != 1) { return services::Status(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); }
    }
    return s;
}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    return inputSize;
}

}// namespace interface1
}// namespace forward
}// namespace reshape
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
