/* file: lcn_layer_forward_batch.h */
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

#include "lcn_layer_forward_types.h"
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
namespace forward
{
namespace interface1
{
/**
 * Allocates memory to store the result of forward  local contrast normalization layer
 * \param[in] input     %Input object for the algorithm
 * \param[in] parameter %Parameter of forward local contrast normalization layer
 * \param[in] method    Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using namespace data_management;
    const Input *in = static_cast<const Input * >(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    const services::Collection<size_t> inDims = in->get(layers::forward::data)->getDimensions();

    if (!get(layers::forward::value))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::value, getValueSize(inDims, parameter, method));
    }

    set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));

    if(!get(auxCenteredData))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(auxCenteredData, inDims);
    }

    if(!get(auxC))
    {
        services::Collection<size_t> cDims;
        getCDimensions(in, algParameter, cDims);

        DAAL_ALLOCATE_TENSOR_AND_SET(auxC, cDims);
    }

    if(!get(auxInvMax))
    {
        services::Collection<size_t> sigmaDims;
        getSigmaDimensions(in, algParameter, sigmaDims);

        DAAL_ALLOCATE_TENSOR_AND_SET(auxInvMax, sigmaDims);
    }

    if(!algParameter->predictionStage)
    {
        if(!get(auxSigma))
        {
            services::Collection<size_t> sigmaDims;
            getSigmaDimensions(in, algParameter, sigmaDims);

            DAAL_ALLOCATE_TENSOR_AND_SET(auxSigma, sigmaDims);
        }
    }
}

}// namespace interface1
}// namespace forward
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
