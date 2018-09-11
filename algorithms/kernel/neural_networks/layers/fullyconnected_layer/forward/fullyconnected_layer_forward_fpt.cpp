/* file: fullyconnected_layer_forward_fpt.cpp */
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
