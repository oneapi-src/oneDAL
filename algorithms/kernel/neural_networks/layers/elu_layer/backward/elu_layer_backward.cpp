/* file: elu_layer_backward.cpp */
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
//  Implementation of ELU calculation algorithm and types methods.
//--
*/

#include "elu_layer_backward_types.h"
#include "elu_layer_types.h"
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
namespace elu
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_ELU_BACKWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for the backward ELU layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(LayerDataId id) const
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    if (!layerData) { return data_management::TensorPtr(); }
    return data_management::Tensor::cast((*layerData)[id]);
}

/**
 * Sets an input object for the backward ELU layer
 * \param[in] id     Identifier of the input object
 * \param[in] value  Pointer to the input object
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData = get(layers::backward::inputFromForward);
    if (layerData) { (*layerData)[id] = value; }
}

/**
 * Checks an input object of the backward ELU layer
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const layers::Parameter *param = static_cast<const layers::Parameter *>(par);
    if (!param->propagateGradient) { return services::Status(); }

    services::Status status;
    DAAL_CHECK_STATUS(status, layers::backward::Input::check(par, method));

    auto &inputGradientDimensions = get(layers::backward::inputGradient)->getDimensions();
    DAAL_CHECK_TENSOR(status, get(elu::auxData).get(), auxDataStr(), &inputGradientDimensions);

    // DAAL_CHECK_TENSOR(status, get(elu::auxIntermediateValue).get(), auxIntermediateValueStr(), &inputGradientDimensions);

    return status;
}

/** \brief Default constructor */
Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward ELU layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input,
                               const daal::algorithms::Parameter *par, int method) const
{
    using namespace daal::services;
    using daal::data_management::Tensor;

    Status status;

    const layers::Parameter *param = static_cast<const layers::Parameter *>(par);
    if (!param->propagateGradient) { return services::Status(); }

    const Input *in = static_cast<const Input *>(input);
    const Tensor *auxDataTensor = in->get(elu::auxData).get();
    DAAL_CHECK(auxDataTensor, Error::create(ErrorNullTensor, ArgumentName, inputGradientStr()));

    auto &auxDataDimensions = auxDataTensor->getDimensions();
    DAAL_CHECK_TENSOR(status, get(layers::backward::gradient).get(), gradientStr(), &auxDataDimensions);

    return status;
}

}// namespace interface1
}// namespace backward
}// namespace elu
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
