/* file: tanh_layer_backward.cpp */
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
//  Implementation of tanh calculation algorithm and types methods.
//--
*/

#include "tanh_layer_backward_types.h"
#include "tanh_layer_types.h"
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
namespace tanh
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_TANH_BACKWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for the backward hyperbolic tangent layer
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
 * Sets an input object for the backward hyperbolic tangent layer
 * \param[in] id     Identifier of the input object
 * \param[in] value  Pointer to the input object
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the backward hyperbolic tangent layer
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(par);
    if (!param->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(par, method));

    const services::Collection<size_t> &inputDimensions = get(layers::backward::inputGradient)->getDimensions();
    const layers::Parameter *parameter = static_cast<const Parameter * >(par);

    if(!parameter->predictionStage)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(tanh::auxValue).get(), auxValueStr(), &inputDimensions));
    }
    return s;
}

    /** \brief Default constructor */
Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward hyperbolic tangent layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(par);
    if (!param->propagateGradient) { return services::Status(); }

    const Input *in = static_cast<const Input *>(input);
    return data_management::checkTensor(
               get(layers::backward::gradient).get(), gradientStr(), &(in->get(tanh::auxValue)->getDimensions()));
}

}// namespace interface1
}// namespace backward
}// namespace tanh
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
