/* file: eltwise_sum_layer_backward_fpt.cpp */
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
//  Implementation of element-wise sum calculation algorithm and types methods.
//--
*/

#include "eltwise_sum_layer_backward_types.h"
#include "eltwise_sum_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
namespace backward
{
namespace interface1
{

using namespace daal::data_management;

/**
 * Allocates memory to store the result of backward element-wise sum layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward element-wise sum layer
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input,
                                              const daal::algorithms::Parameter *parameter,
                                              const int method)
{
    services::Status s;

    const Input *eltwiseInput = dynamic_cast<const Input *>(input);
    DAAL_CHECK(eltwiseInput, services::ErrorNullInput);
    DAAL_CHECK_STATUS(s, eltwiseInput->check(parameter, method));

    const size_t nOutputs     = eltwiseInput->getNumberOfCoefficients();
    TensorPtr auxCoefficients = eltwiseInput->get(eltwise_sum::auxCoefficients);
    TensorPtr inputGradient   = eltwiseInput->get(layers::backward::inputGradient);

    // We call this function just to allocate
    // resultLayerData and consciously ignore returned value
    LayerDataPtr resultLayerData = getResultLayerDataAllocateIfEmpty();
    DAAL_CHECK_MALLOC(resultLayerData);

    return allocateNewOutputTensors<algorithmFPType>(inputGradient, nOutputs);
}

template <typename algorithmFPType>
services::Status Result::allocateNewOutputTensors(const TensorPtr &inputGradient, size_t nOutputs)
{
    const services::Collection<size_t> &inputGradientDimensions = inputGradient->getDimensions();
    services::Status s;
    for (size_t i = 0; i < nOutputs; i++)
    {
        TensorPtr resultLayerDataI = HomogenTensor<algorithmFPType>::create(inputGradientDimensions, Tensor::doAllocate, &s);
        DAAL_CHECK_STATUS_VAR(s);

        set(layers::backward::resultLayerData, resultLayerDataI, i);
    }

    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(
    const daal::algorithms::Input *, const daal::algorithms::Parameter *, const int);

template DAAL_EXPORT services::Status Result::allocateNewOutputTensors<DAAL_FPTYPE>(const TensorPtr &, size_t);

}// namespace interface1
}// namespace backward
}// namespace eltwise_sum
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
