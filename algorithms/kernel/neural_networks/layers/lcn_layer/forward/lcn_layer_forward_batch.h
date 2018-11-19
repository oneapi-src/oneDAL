/* file: lcn_layer_forward_batch.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using namespace data_management;
    const Input *in = static_cast<const Input * >(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    const services::Collection<size_t> inDims = in->get(layers::forward::data)->getDimensions();
    services::Status s;
    if (!get(layers::forward::value))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, getValueSize(inDims, parameter, method));
    }

    set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));

    if(!get(auxCenteredData))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, auxCenteredData, inDims);
    }

    if(!get(auxC))
    {
        services::Collection<size_t> cDims;
        getCDimensions(in, algParameter, cDims);

        DAAL_ALLOCATE_TENSOR_AND_SET(s, auxC, cDims);
    }

    if(!get(auxInvMax))
    {
        services::Collection<size_t> sigmaDims;
        getSigmaDimensions(in, algParameter, sigmaDims);

        DAAL_ALLOCATE_TENSOR_AND_SET(s, auxInvMax, sigmaDims);
    }

    if(!algParameter->predictionStage)
    {
        if(!get(auxSigma))
        {
            services::Collection<size_t> sigmaDims;
            getSigmaDimensions(in, algParameter, sigmaDims);

            DAAL_ALLOCATE_TENSOR_AND_SET(s, auxSigma, sigmaDims);
        }
    }
    return services::Status();
}

}// namespace interface1
}// namespace forward
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
