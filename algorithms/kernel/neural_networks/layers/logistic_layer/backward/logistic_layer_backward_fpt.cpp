/* file: logistic_layer_backward_fpt.cpp */
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
//  Implementation of logistic calculation algorithm and types methods.
//--
*/

#include "logistic_layer_backward_types.h"
#include "logistic_layer_types.h"

#include "service_mkl_tensor.h"
#include "tensor.h"
#include "service_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace logistic
{
namespace backward
{
namespace interface1
{
/**
* Allocates memory to store the result of the backward logistic layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the algorithm
 * \param[in] par %Parameter of the backward logistic layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const layers::Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }
    services::Status s;
    if (!get(layers::backward::gradient))
    {
        const Input *in = static_cast<const Input *>(input);

        data_management::TensorPtr valueTable = in->get(auxValue);

        if (!valueTable) return services::Status(services::ErrorNullInputNumericTable);

        auto inTensor = in->get(layers::backward::inputGradient);
        data_management::HomogenTensor<algorithmFPType> *inHomo = dynamic_cast<data_management::HomogenTensor<algorithmFPType>*>( inTensor.get() );
        internal       ::MklTensor    <algorithmFPType> *inMkl  = dynamic_cast<internal       ::MklTensor    <algorithmFPType>*>( inTensor.get() );

        if(inHomo || inMkl)
        {
            set(layers::backward::gradient, inTensor);
        }
        else
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, valueTable->getDimensions());
        }
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace logistic
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
