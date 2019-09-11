/* file: fullyconnected_layer_backward.cpp */
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
//  Implementation of fullyconnected calculation algorithm and types methods.
//--
*/

#include "fullyconnected_layer_backward_types.h"
#include "fullyconnected_layer_types.h"
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
namespace fullyconnected
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_FULLYCONNECTED_BACKWARD_RESULT_ID);
/**
 * Default constructor
 */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for backward fully-connected layer
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
 * Sets input for the backward fully-connected layer
 * \param[in] id    Identifier of the input  object
 * \param[in] value Input object to set
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the fully-connected layer
 * \param[in] par       %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(par, method));

    const Parameter *param = static_cast<const Parameter*>(par);

    data_management::TensorPtr xTensor = get(auxData);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(xTensor.get(), auxDataStr()));

    const services::Collection<size_t> &xDims = xTensor->getDimensions();
    const services::Collection<size_t> &gDims = get(layers::backward::inputGradient)->getDimensions();

    services::Collection<size_t> gradDims;
    gradDims.push_back(xDims[0]);
    gradDims.push_back(param->nOutputs);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::inputGradient).get(), inputGradientStr(), &gradDims));

    if (param->propagateGradient)
    {
        services::Collection<size_t> wDims = xDims;
        wDims[0] = param->nOutputs;
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxWeights).get(), auxWeightsStr(), &wDims));
    }
    return s;
}

/**
 * Default constructor
 */
Result::Result() : layers::backward::Result() {}

/**
 * Checks the result of the fully-connected layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Result::check(input, par, method));

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(par);

    if (param->propagateGradient)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &(algInput->get(auxData)->getDimensions())));
    }
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::weightDerivatives).get(), weightDerivativesStr(), &(algInput->get(auxWeights)->getDimensions())));

    services::Collection<size_t> bDims;
    bDims.push_back(param->nOutputs);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::biasDerivatives).get(), biasDerivativesStr(), &bDims));
    return s;
}

}// namespace interface1
}// namespace backward
}// namespace fullyconnected
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
