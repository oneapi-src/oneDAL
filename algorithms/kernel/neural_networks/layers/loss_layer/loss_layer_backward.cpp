/* file: loss_layer_backward.cpp */
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
//  Implementation of loss calculation algorithm and types methods.
//--
*/

#include "loss_layer_backward_types.h"
#include "daal_strings.h"

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
namespace backward
{
namespace interface1
{
/** Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Checks an input object for the backward loss layer
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const layers::Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) return services::Status();

    if(Argument::size() != 2) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    if (!layerData) return services::Status(services::ErrorNullLayerData);
    return services::Status();
}

    /** Default constructor */
Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward loss layer
 * \param[in] input   %Input object for the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    return services::Status();
}

}// namespace interface1
}// namespace backward
}// namespace loss
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
