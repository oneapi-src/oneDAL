/* file: lrn_layer_forward.cpp */
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
//  Implementation of lrn calculation algorithm and types methods.
//--
*/

#include "lrn_layer_forward_types.h"
#include "lrn_layer_types.h"
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
namespace lrn
{
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_LRN_FORWARD_RESULT_ID);
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

/** \brief Constructor */
Result::Result() : layers::forward::Result() {};

/**
 * Returns the result of the forward local response normalization layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
data_management::TensorPtr Result::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(!layerData)
        return data_management::TensorPtr();
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of the forward local response normalization layer
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(layerData)
        (*layerData)[id] = value;
}

/**
 * Checks the result of the forward local response normalization layer
 * \param[in] input   %Input object for the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, par, method));
    const Parameter *param = static_cast<const Parameter *>(par);

    data_management::TensorPtr dataTable = (static_cast<const layers::forward::Input *>(input))->get(
                                                                 layers::forward::data);
    data_management::TensorPtr resultTable = get(layers::forward::value);
    data_management::NumericTablePtr dimensionTable = param->dimension;

    data_management::BlockDescriptor<int> block;
    dimensionTable->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *dataInt = block.getBlockPtr();
    size_t dim = dataInt[0];
    if(dim >= dataTable->getNumberOfDimensions())
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, dimensionStr()));
    }
    dimensionTable->releaseBlockOfRows(block);

    if (!dataTable)   return services::Status(services::ErrorNullInputNumericTable);
    if (!resultTable) return services::Status(services::ErrorNullOutputNumericTable);

    const services::Collection<size_t> &dataDims = dataTable->getDimensions();
    if (!param->predictionStage)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxData).get(), auxDataStr(), &dataDims));
    }
    return data_management::checkTensor(get(auxSmBeta).get(), auxSmBetaStr(), &dataDims);
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

/**
 * Sets the result that is used in backward local response normalization layer
 * \param[in] input     Pointer to an object containing the input data
 */
services::Status Result::setResultForBackward(const daal::algorithms::Input *input)
{
    const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
    set(lrn::auxData, in->get(layers::forward::data));
    return services::Status();
}

}// namespace interface1
}// namespace forward
}// namespace lrn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
