/* file: convolution2d_layer_backward_batch.h */
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
//  Implementation of onvolution2d calculation algorithm and types methods.
//--
*/

#include "convolution2d_layer_backward_types.h"
#include "convolution2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace convolution2d
{
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of backward 2D convolution layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward 2D convolution layer
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::HomogenTensor;

    const Input *in = static_cast<const Input *>(input);
    const Parameter *param =  static_cast<const Parameter * >(parameter);

    services::Collection<size_t> bDims;
    bDims.push_back(param->nKernels);

    services::SharedPtr<Tensor> valueTable = in->get(auxData);
    services::SharedPtr<Tensor> wTable     = in->get(auxWeights);

    if(valueTable == 0 || wTable == 0) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

    if (!get(layers::backward::gradient))
    {
        set(layers::backward::gradient, services::SharedPtr<Tensor>(
                      new HomogenTensor<algorithmFPType>(valueTable->getDimensions(), Tensor::doAllocate)));
    }
    if (!get(layers::backward::weightDerivatives))
    {
        set(layers::backward::weightDerivatives, services::SharedPtr<Tensor>(
                          new HomogenTensor<algorithmFPType>(wTable->getDimensions(), Tensor::doAllocate)));
    }
    if (!get(layers::backward::biasDerivatives))
    {
        set(layers::backward::biasDerivatives, services::SharedPtr<Tensor>(
                          new HomogenTensor<algorithmFPType>(bDims, Tensor::doAllocate)));
    }
}

}// namespace interface1
}// namespace forward
}// namespace convolution2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
