/* file: softmax_layer_backward.cpp */
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
//  Implementation of softmax calculation algorithm and types methods.
//--
*/

#include "softmax_layer_backward_types.h"
#include "softmax_layer_types.h"
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
namespace softmax
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_SOFTMAX_BACKWARD_RESULT_ID);
/** Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
* Returns an input object for the backward softmax layer
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
data_management::TensorPtr Input::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets an input object for the backward softmax layer
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    (*layerData)[id] = value;
}

/**
* Checks input object for the backward softmax layer
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const softmax::Parameter *parameter = static_cast<const softmax::Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(par, method));

    const services::Collection<size_t>& inputDimensions = get(layers::backward::inputGradient)->getDimensions();

    if(!parameter->predictionStage)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxValue).get(), auxValueStr(), &inputDimensions));
    }

    DAAL_CHECK_EX(parameter->dimension < inputDimensions.size(), services::ErrorIncorrectParameter, services::ParameterName, dimensionStr());
    return s;
}

Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward softmax layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const softmax::Parameter *parameter = static_cast<const softmax::Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    const Input *in = static_cast<const Input *>(input);
    return data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &(in->get(auxValue)->getDimensions()));
}

}// namespace interface1
}// namespace backward
}// namespace softmax
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
