/* file: maximum_pooling1d_layer_backward.cpp */
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
//  Implementation of maximum_pooling1d calculation algorithm and types methods.
//--
*/

#include "maximum_pooling1d_layer_backward_types.h"
#include "maximum_pooling1d_layer_types.h"
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
namespace maximum_pooling1d
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_MAXIMUM_POOLING1D_BACKWARD_RESULT_ID);
/**
 * Default constructor
 */
Input::Input() {}
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for backward maximum 1D pooling layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(LayerDataId id) const
{
    layers::LayerDataPtr inputData = get(layers::backward::inputFromForward);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*inputData)[id]);
}

/**
 * Returns an input object for backward maximum 1D pooling layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::NumericTablePtr Input::get(LayerDataNumericTableId id) const
{
    layers::LayerDataPtr inputData = get(layers::backward::inputFromForward);
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*inputData)[id]);
}

/**
 * Sets an input object for the backward maximum 1D pooling layer
 * \param[in] id  Identifier of the input object
 * \param[in] ptr Pointer to the object
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &ptr)
{
    layers::LayerDataPtr inputData = get(layers::backward::inputFromForward);
    (*inputData)[id] = ptr;
}

/**
 * Sets an input object for the backward maximum 1D pooling layer
 * \param[in] id  Identifier of the input object
 * \param[in] ptr Pointer to the object
 */
void Input::set(LayerDataNumericTableId id, const data_management::NumericTablePtr &ptr)
{
    layers::LayerDataPtr inputData = get(layers::backward::inputFromForward);
    (*inputData)[id] = ptr;
}

/**
 * Checks an input object for the backward maximum 1D pooling layer
 * \param[in] parameter Algorithm parameter
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, pooling1d::backward::Input::check(parameter, method));

    data_management::NumericTablePtr auxInputDimensions = get(maximum_pooling1d::auxInputDimensions);
    const services::Collection<size_t> &inputGradDims = get(layers::backward::inputGradient)->getDimensions();
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(maximum_pooling1d::auxSelectedIndices).get(), auxSelectedIndicesStr(), &inputGradDims));
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(auxInputDimensions.get(), auxInputDimensionsStr(), data_management::packed_mask, 0, inputGradDims.size(), 1));
    return s;
}

data_management::NumericTablePtr Input::getAuxInputDimensions() const
{
    return get(auxInputDimensions);
}

/** Default constructor */
Result::Result() : pooling1d::backward::Result() {};

/**
 * Checks the result of the backward maximum 1D pooling layer
 * \param[in] input     %Input object for the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    return pooling1d::backward::Result::check(input, parameter, method);
}

}// namespace interface1
}// namespace backward
}// namespace maximum_pooling1d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
