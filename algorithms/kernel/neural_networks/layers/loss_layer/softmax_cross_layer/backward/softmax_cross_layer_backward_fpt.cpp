/* file: softmax_cross_layer_backward_fpt.cpp */
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
//  Implementation of softmax cross calculation algorithm and types methods.
//--
*/

#include "softmax_cross_layer_types.h"
#include "softmax_cross_layer_backward_types.h"

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
/**
 * Allocates memory to store the result of the backward softmax cross-entropy layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the layer
 * \param[in] parameter %Parameter of the backward softmax cross-entropy layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const layers::Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    services::Status s;
    if (!get(layers::backward::gradient))
    {
        const Input *in = static_cast<const Input *>(input);

        data_management::TensorPtr probabilitiesTable = in->get(auxProbabilities);

        if(probabilitiesTable == 0) return services::Status(services::ErrorNullInputNumericTable);

        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, probabilitiesTable->getDimensions());
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace softmax_cross
}// namespace loss
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
