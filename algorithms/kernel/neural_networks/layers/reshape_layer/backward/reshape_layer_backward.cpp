/* file: reshape_layer_backward.cpp */
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
//  Implementation of reshape calculation algorithm and types methods.
//--
*/

#include "reshape_layer_backward_types.h"
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
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_RESHAPE_BACKWARD_RESULT_ID);
/** Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
* Returns an input object for the backward reshape layer
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets an input object for the backward reshape layer
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(LayerDataId id, const data_management::NumericTablePtr &value)
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    (*layerData)[id] = value;
}

/**
* Checks input object for the backward reshape layer
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method
*/
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(par, method));
    return s;
}

Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward reshape layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Input *in = static_cast<const Input *>(input);
    const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);

    if (!parameter->propagateGradient) { return services::Status(); }

    data_management::NumericTablePtr dimsTable = in->get(layers::reshape::auxInputDimensions);

    size_t nDims = dimsTable->getNumberOfColumns();

    services::Collection<size_t> iDims( nDims );

    data_management::BlockDescriptor<int> block;
    dimsTable->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *dataArray = block.getBlockPtr();

    for(size_t i = 0; i < nDims; i++)
    {
        iDims[i] = dataArray[i];
    }

    dimsTable->releaseBlockOfRows(block);

    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::gradient).get(), resultLayerDataStr(), &iDims));
    return s;
}

}// namespace interface1
}// namespace backward
}// namespace reshape
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
