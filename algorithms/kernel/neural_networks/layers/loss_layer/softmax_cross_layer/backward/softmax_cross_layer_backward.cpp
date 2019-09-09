/* file: softmax_cross_layer_backward.cpp */
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
//  Implementation of softmax cross calculation algorithm and types methods.
//--
*/

#include "softmax_cross_layer_types.h"
#include "softmax_cross_layer_backward_types.h"
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
namespace loss
{
namespace softmax_cross
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOSS_SOFTMAX_CROSS_BACKWARD_RESULT_ID);
/** Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for the backward softmax cross-entropy layer
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
 * Sets an input object for the backward softmax cross-entropy layer
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
 * Checks an input object for the backward softmax cross-entropy layer
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const layers::Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, loss::backward::Input::check(par, method));

    data_management::TensorPtr auxProbabilitiesTensor = get(auxProbabilities);
    data_management::TensorPtr auxGroundTruthTensor = get(auxGroundTruth);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(auxProbabilitiesTensor.get(), auxProbabilitiesStr()));

    const layers::loss::softmax_cross::Parameter *param = static_cast<const layers::loss::softmax_cross::Parameter * >(par);
    size_t dim = param->dimension;

    services::Collection<size_t> groundTruthDims = auxProbabilitiesTensor->getDimensions();
    DAAL_CHECK_EX(dim <= groundTruthDims.size() - 1 && dim != 0, services::ErrorIncorrectParameter, services::ParameterName, dimensionStr());

    groundTruthDims[dim] = 1;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(auxGroundTruthTensor.get(), auxGroundTruthStr(), &groundTruthDims));
    return s;
}

/** Default constructor */
Result::Result() : loss::backward::Result() {};

/**
 * Checks the result of the backward softmax cross-entropy layer
 * \param[in] input   %Input object for the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const layers::Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    const Input *algInput = static_cast<const Input *>(input);
    //get expected gradient dimensions
    const services::Collection<size_t> &gradDims = algInput->get(auxProbabilities)->getDimensions();
    return data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &gradDims);
}

}// namespace interface1
}// namespace backward
}// namespace softmax_cross
}// namespace loss
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
