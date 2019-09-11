/* file: prelu_layer_forward.cpp */
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
//  Implementation of prelu calculation algorithm and types methods.
//--
*/

#include "prelu_layer_forward_types.h"
#include "prelu_layer_types.h"
#include "serialization_utils.h"
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
namespace prelu
{
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_PRELU_FORWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
* Returns dimensions of weights tensor
* \return Dimensions of weights tensor
*/
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *par) const
{
    const Parameter *parameter =  static_cast<const Parameter *>(par);
    data_management::TensorPtr dataTensor = get(layers::forward::data);

    size_t wStartDim = parameter->dataDimension;
    size_t wDimNumber = parameter->weightsDimension;

    services::Collection<size_t> _dims = dataTensor->getDimensions();
    services::Collection<size_t> _wdims;

    for (size_t i = wStartDim; i < wStartDim + wDimNumber; i++)
    {
        _wdims.push_back(_dims[i]);
    }
    return _wdims;
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
 * Checks input object of the forward prelu layer
 * \param[in] par     Layer parameter
 * \param[in] method  Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Input::check(par, method));
    const Parameter *parameter =  static_cast<const Parameter *>(par);
    const services::Collection<size_t> &dataDimensions = get(layers::forward::data)->getDimensions();
    DAAL_CHECK_EX(parameter->dataDimension <= dataDimensions.size() - parameter->weightsDimension, services::ErrorIncorrectParameter,
                  services::ParameterName, dataDimensionStr());
    DAAL_CHECK_EX(parameter->weightsDimension != 0, services::ErrorIncorrectParameter, services::ParameterName, weightsDimensionStr());
    const services::Collection<size_t> weightsDimensions = getWeightsSizes(parameter);
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::forward::weights).get(), weightsStr(), &weightsDimensions));
    return s;
}

/** \brief Default constructor */
Result::Result() : layers::forward::Result() {};

/**
 * Returns the result of the forward prelu layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
data_management::TensorPtr Result::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of the forward prelu layers
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    (*layerData)[id] = value;
}

/**
 * Checks the result of the forward prelu layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, par, method));

    const Input *in = static_cast<const Input * >(input);
    const Parameter *parameter =  static_cast<const Parameter *>(par);
    const services::Collection<size_t> weightsDimension = in->getWeightsSizes(parameter);
    const services::Collection<size_t> &inputDimensions = in->get(layers::forward::data)->getDimensions();

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::forward::value).get(), valueStr(), &inputDimensions));
    if(!parameter->predictionStage)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(prelu::auxData).get(), auxDataStr(), &inputDimensions));
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(prelu::auxWeights).get(), auxWeightsStr(), &weightsDimension));
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

/**
 * Sets the result that is used in backward prelu layer
 * \param[in] input     Pointer to an object containing the input data
 */
services::Status Result::setResultForBackward(const daal::algorithms::Input *input)
{
    const Input *in = static_cast<const Input *>(input);
    set(auxData, in->get(layers::forward::data));
    set(auxWeights, in->get(layers::forward::weights));
    return services::Status();
}

}// namespace interface1
}// namespace forward
}// namespace prelu
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
