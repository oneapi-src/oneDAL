/* file: dropout_layer_forward_fpt.cpp */
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
//  Implementation of dropout calculation algorithm and types methods.
//--
*/

#include "dropout_layer_forward_types.h"
#include "dropout_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace dropout
{
namespace forward
{
namespace interface1
{
/**
 * Allocates memory to store the result of the forward dropout layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the layer
 * \param[in] parameter %Parameter of the forward dropout layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    services::Status s;
    if (!get(layers::forward::value))
    {
        if (par->predictionStage)
        {
            set(layers::forward::value, in->get(layers::forward::data));
        }
        else
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, in->get(layers::forward::data)->getDimensions());
        }
    }

    if(!par->predictionStage)
    {
        if (!get(layers::forward::resultForBackward))
        {
            set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
        }
        if (!get(dropout::auxRetainMask))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, dropout::auxRetainMask, in->get(layers::forward::data)->getDimensions());
        }
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace dropout
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
