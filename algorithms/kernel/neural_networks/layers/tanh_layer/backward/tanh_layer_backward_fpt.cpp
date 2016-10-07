/* file: tanh_layer_backward_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of tanh calculation algorithm and types methods.
//--
*/

#include "tanh_layer_backward_types.h"
#include "tanh_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace tanh
{
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of the backward hyperbolic tangent layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the algorithm
 * \param[in] par %Parameter of the backward layer
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const Parameter *param = static_cast<const Parameter *>(par);
    if (!param->propagateGradient) { return; }

    if (!get(layers::backward::gradient))
    {
        const Input *in = static_cast<const Input *>(input);

        services::SharedPtr<data_management::Tensor> valueTable = in->get(auxValue);

        if (!valueTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

        DAAL_ALLOCATE_TENSOR_AND_SET(layers::backward::gradient, valueTable->getDimensions());
    }
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace tanh
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
