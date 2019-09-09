/* file: prelu_layer_backward_fpt.cpp */
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
//  Implementation of prelu calculation algorithm and types methods.
//--
*/

#include "prelu_layer_backward_types.h"
#include "prelu_layer_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace backward
{
namespace interface1
{
/**
* Allocates memory to store the result of the backward prelu layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the algorithm
 * \param[in] par %Parameter of the backward prelu layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const Input *in = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(par);

    data_management::TensorPtr valueTensor = in->get(auxData);
    data_management::TensorPtr weightsTensor = in->get(auxWeights);

    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(valueTensor.get(), auxDataStr()));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(weightsTensor.get(), auxWeightsStr()));

    if (param->propagateGradient && !get(layers::backward::gradient))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, valueTensor->getDimensions());
    }
    if (!get(layers::backward::weightDerivatives))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::weightDerivatives, weightsTensor->getDimensions());
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace prelu
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
