/* file: softmax_layer_backward_fpt.cpp */
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
//  Implementation of softmax calculation algorithm and types methods.
//--
*/

#include "softmax_layer_backward_types.h"
#include "softmax_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace softmax
{
namespace backward
{
namespace interface1
{
/**
* Allocates memory to store the result of the backward softmax layer
* \param[in] input     Pointer to an object containing the input data
* \param[in] parameter %Parameter of the backward softmax layer
* \param[in] method    Computation method
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const softmax::Parameter *param = static_cast<const softmax::Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    const Input *in = static_cast<const Input *>(input);

    data_management::TensorPtr valueTable = in->get(auxValue);
    services::Status s;
    if(valueTable == 0) return services::Status(services::ErrorNullInputNumericTable);

    if (!get(layers::backward::gradient))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, valueTable->getDimensions());
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace softmax
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
