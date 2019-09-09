/* file: fullyconnected_layer_forward_fpt.cpp */
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
//  Implementation of fullyconnected calculation algorithm and types methods.
//--
*/

#include "fullyconnected_layer_forward_types.h"
#include "fullyconnected_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
namespace forward
{
namespace interface1
{
/**
* Allocates memory to store the result of forward  fully-connected layer
 * \param[in] parameter %Parameter of forward fully-connected layer
 * \param[in] method    Computation method for the layer
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Input::allocate(const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::services::SharedPtr;
    using daal::data_management::Tensor;
    using daal::data_management::HomogenTensor;

    const Parameter *param =  static_cast<const Parameter *>(parameter);

    services::Status s;

    if( !get(layers::forward::weights) )
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::weights, getWeightsSizes(param));
    }

    if( !get(layers::forward::biases) )
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::biases, getBiasesSizes(param));
    }
    return s;
}
/**
 * Allocates memory to store the result of forward  fully-connected layer
 * \param[in] input     %Input object for the algorithm
 * \param[in] parameter %Parameter of forward fully-connected layer
 * \param[in] method    Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input * >(input);
    services::Status s;
    if (!get(layers::forward::value))
    {
        const services::Collection<size_t> &valueDims = getValueSize(in->get(layers::forward::data)->getDimensions(), parameter, method);
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, valueDims);
    }

    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        if (!get(layers::forward::resultForBackward))
        {
            set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
        }
        DAAL_CHECK_STATUS(s, setResultForBackward(input));
    }
    return s;
}

template DAAL_EXPORT services::Status Input::allocate<DAAL_FPTYPE>(const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace fullyconnected
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
