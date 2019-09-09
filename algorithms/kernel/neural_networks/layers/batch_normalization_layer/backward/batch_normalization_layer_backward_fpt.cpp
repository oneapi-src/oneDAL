/* file: batch_normalization_layer_backward_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
