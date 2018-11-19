/* file: batch_normalization_layer_backward_fpt.cpp */
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
//  Implementation of batch_normalization calculation algorithm and types methods.
//--
*/

#include "batch_normalization_layer_backward_types.h"
#include "batch_normalization_layer_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of the backward batch normalization layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of the backward batch normalization layer
 * \param[in] method Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr inputGradientTensor = in->get(layers::backward::inputGradient);

    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(inputGradientTensor.get(), inputGradientStr()));

    if (algParameter->propagateGradient)
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, inputGradientTensor->getDimensions());
    }

    size_t dimension = algParameter->dimension;
    size_t dimensionSize = inputGradientTensor->getDimensionSize(dimension);
    services::Collection<size_t> requiredDimensionSizes(1);
    requiredDimensionSizes[0] = dimensionSize;

    if (!get(layers::backward::weightDerivatives))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::weightDerivatives, requiredDimensionSizes);
    }
    if (!get(layers::backward::biasDerivatives))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::biasDerivatives, requiredDimensionSizes);
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace batch_normalization
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
