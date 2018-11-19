/* file: average_pooling1d_layer_forward.cpp */
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
//  Implementation of average_pooling1d calculation algorithm and types methods.
//--
*/

#include "average_pooling1d_layer_forward_types.h"
#include "average_pooling1d_layer_types.h"
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
namespace average_pooling1d
{
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING1D_FORWARD_RESULT_ID);
/**
 * Default constructor
 */
Result::Result() {}

/**
 * Returns the result of the forward average 1D pooling layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData = get(layers::forward::resultForBackward);
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of the forward average 1D pooling layer
 * \param[in] id      Identifier of the result
 * \param[in] ptr     Result
 */
void Result::set(LayerDataId id, const data_management::NumericTablePtr &ptr)
{
    layers::LayerDataPtr layerData = get(layers::forward::resultForBackward);
    (*layerData)[id] = ptr;
}

/**
 * Checks the result of the forward average 1D pooling layer
 * \param[in] input     %Input of the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method of the layer
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, pooling1d::forward::Result::check(input, parameter, method));

    const Parameter *param =  static_cast<const Parameter *>(parameter);

    if(!param->predictionStage)
    {
        const Input *in = static_cast<const Input *>(input);

        const services::Collection<size_t> &dataDims = in->get(layers::forward::data)->getDimensions();

        data_management::NumericTablePtr auxInputDimensions = get(average_pooling1d::auxInputDimensions);
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(auxInputDimensions.get(), auxInputDimensionsStr(), data_management::packed_mask, 0, dataDims.size(), 1));
    }
    return s;
}

}// namespace interface1
}// namespace forward
}// namespace average_pooling1d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
