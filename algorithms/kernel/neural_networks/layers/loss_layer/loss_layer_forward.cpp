/* file: loss_layer_forward.cpp */
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
namespace forward
{
namespace interface1
{
/** Default constructor */
Input::Input(size_t nElements) : layers::forward::Input(nElements) {};

/**
 * Returns input Tensor of the loss layer algorithm
 * \param[in] id    Identifier of the input tensor
 * \return          %Input tensor that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(forward::InputId id) const
{
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the loss layer algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::TensorPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Returns dimensions of weights tensor
 * \return Dimensions of weights tensor
 */
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}

/**
 * Returns dimensions of biases tensor
 * \return Dimensions of biases tensor
 */
const services::Collection<size_t> Input::getBiasesSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}

   /** Default constructor */
Result::Result() : layers::forward::Result() {};

/**
 * Checks the result of the forward loss layer
 * \param[in] input   %Input object for the loss layer
 * \param[in] par     %Parameter of the loss layer
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Input *in = static_cast<const Input *>(input);
    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(in->get(layers::forward::data).get(), dataStr()));

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::forward::value).get(), valueStr()));
    return s;
}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
        const daal::algorithms::Parameter *par, const int method) const
{
    return inputSize;
}

}// namespace interface1
}// namespace forward
}// namespace loss
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
