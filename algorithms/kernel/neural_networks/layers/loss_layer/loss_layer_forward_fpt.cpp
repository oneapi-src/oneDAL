/* file: loss_layer_forward_fpt.cpp */
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

#include "loss_layer_forward_types.h"

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
namespace forward
{
namespace interface1
{
/**
 * Allocates memory to store the result of the forward loss layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the loss layer
 * \param[in] parameter %Parameter of the forward loss layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);

    services::Collection<size_t> valueDim(1);
    valueDim[0] = 1;
    services::Status s;
    DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, valueDim);
    Argument::set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace loss
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
