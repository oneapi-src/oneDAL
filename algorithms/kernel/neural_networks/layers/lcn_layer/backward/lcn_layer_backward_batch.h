/* file: lcn_layer_backward_batch.h */
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
//  Implementation of lcn calculation algorithm and types methods.
//--
*/

#include "lcn_layer_backward_types.h"
#include "lcn_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lcn
{
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of backward local contrast normalization layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward local contrast normalization layer
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::HomogenTensor;

    const Input *in = static_cast<const Input *>(input);
    const Parameter *param =  static_cast<const Parameter * >(parameter);

    services::SharedPtr<Tensor> centeredDataTensor = in->get(auxCenteredData);
    services::SharedPtr<Tensor> sigmaTensor        = in->get(auxSigma);
    services::SharedPtr<Tensor> cTensor            = in->get(auxC);
    services::SharedPtr<Tensor> maxTensor          = in->get(auxInvMax);

    if(centeredDataTensor == 0 || sigmaTensor == 0 || cTensor == 0 || maxTensor == 0) { this->_errors->add(services::ErrorNullTensor); return; }

    if (!get(layers::backward::gradient))
    {
        set(layers::backward::gradient, services::SharedPtr<Tensor>(
                      new HomogenTensor<algorithmFPType>(centeredDataTensor->getDimensions(), Tensor::doAllocate)));
    }
}

}// namespace interface1
}// namespace forward
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
