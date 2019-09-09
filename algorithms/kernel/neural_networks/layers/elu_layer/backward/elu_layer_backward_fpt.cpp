/* file: elu_layer_backward_fpt.cpp */
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
//  Implementation of ELU calculation algorithm and types methods.
//--
*/

#include "elu_layer_backward_types.h"
#include "elu_layer_types.h"

#include "daal_strings.h"
#include "service_mkl_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace backward
{
namespace interface1
{

using namespace daal::services;
using namespace daal::data_management;

/**
* Allocates memory to store the result of the backward ELU layer
 * \param[in] input     Pointer to an object containing the input data
 * \param[in] method    Computation method for the algorithm
 * \param[in] parameter %Parameter of the backward ELU layer
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input,
                                    const daal::algorithms::Parameter *parameter,
                                    const int method)
{
    Status status;

    auto *in  = static_cast<const Input *>(input);
    auto *par = static_cast<const Parameter *>(parameter);
    if (!par->propagateGradient) { return services::Status(); }

    const Tensor *inputGradientTensor = in->get(layers::backward::inputGradient).get();
    DAAL_CHECK(inputGradientTensor, Error::create(ErrorNullTensor, ArgumentName, inputGradientStr()));

    if (!get(layers::backward::gradient))
    {
        using daal::internal::createTensorKeepingType;
        const TensorPtr gradientTensor = createTensorKeepingType<algorithmFPType>(inputGradientTensor, status);

        DAAL_CHECK_STATUS_VAR(status);
        set(layers::backward::gradient, gradientTensor);
    }
    return services::Status();
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                          const daal::algorithms::Parameter *parameter,
                                                          const int method);

} // namespace interface1
} // namespace backward
} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
