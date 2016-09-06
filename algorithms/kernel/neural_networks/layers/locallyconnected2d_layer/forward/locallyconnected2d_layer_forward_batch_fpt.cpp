/* file: locallyconnected2d_layer_forward_batch_fpt.cpp */
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
//  Implementation of locally connected calculation algorithm and types methods.
//--
*/
#include "locallyconnected2d_layer_forward_types.h"
#include "locallyconnected2d_layer_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace locallyconnected2d
{
namespace forward
{
namespace interface1
{
/**
* Allocates memory to store the result of forward  2D locally connected layer
 * \param[in] parameter %Parameter of forward 2D locally connected layer
 * \param[in] method    Computation method for the layer
*/
template <typename algorithmFPType>
void DAAL_EXPORT Input::allocate(const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::services::SharedPtr;
    using daal::data_management::Tensor;
    using daal::data_management::HomogenTensor;

    const Parameter *param =  static_cast<const Parameter * >(parameter);

    if( !get(layers::forward::weights) )
    {
        TensorPtr tensor(new HomogenTensor<float>(getWeightsSizes(param), Tensor::doAllocate));
        set(layers::forward::weights, tensor);
    }

    if( !get(layers::forward::biases) )
    {
        TensorPtr tensor(new HomogenTensor<float>(getBiasesSizes(param), Tensor::doAllocate));
        set(layers::forward::biases, tensor);
    }
}

/**
 * Allocates memory to store the result of forward  2D locally connected layer
 * \param[in] input     %Input object for the algorithm
 * \param[in] parameter %Parameter of forward 2D locally connected layer
 * \param[in] method    Computation method for the layer
 */
template <typename algorithmFPType>
void DAAL_EXPORT Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using namespace data_management;
    const Input *in = static_cast<const Input * >(input);

    const services::Collection<size_t> &inDims = in->get(layers::forward::data)->getDimensions();

    if (!get(layers::forward::value))
    {
        set(layers::forward::value, TensorPtr(
                      new HomogenTensor<algorithmFPType>(getValueSize(inDims, parameter, method), Tensor::doAllocate)));
    }

    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
        setResultForBackward(input);
    }
}

template DAAL_EXPORT void Input::allocate<DAAL_FPTYPE>(const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace locallyconnected2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
