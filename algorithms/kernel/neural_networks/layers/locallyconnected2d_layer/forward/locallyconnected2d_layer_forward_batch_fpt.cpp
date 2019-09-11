/* file: locallyconnected2d_layer_forward_batch_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
services::Status DAAL_EXPORT Input::allocate(const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::services::SharedPtr;
    using daal::data_management::Tensor;
    using daal::data_management::HomogenTensor;

    const Parameter *param =  static_cast<const Parameter * >(parameter);
    services::Status s;
    if( !get(layers::forward::weights) )
    {
        TensorPtr tensor = HomogenTensor<algorithmFPType>::create(getWeightsSizes(param), Tensor::doAllocate, &s);
        set(layers::forward::weights, tensor);
    }

    if( !get(layers::forward::biases) )
    {
        TensorPtr tensor = HomogenTensor<algorithmFPType>::create(getBiasesSizes(param), Tensor::doAllocate, &s);
        set(layers::forward::biases, tensor);
    }
    return s;
}

/**
 * Allocates memory to store the result of forward  2D locally connected layer
 * \param[in] input     %Input object for the algorithm
 * \param[in] parameter %Parameter of forward 2D locally connected layer
 * \param[in] method    Computation method for the layer
 */
template <typename algorithmFPType>
services::Status DAAL_EXPORT Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using namespace data_management;
    const Input *in = static_cast<const Input * >(input);

    services::Status s;
    const services::Collection<size_t> &inDims = in->get(layers::forward::data)->getDimensions();

    if (!get(layers::forward::value))
    {
        set(layers::forward::value, HomogenTensor<algorithmFPType>::create(getValueSize(inDims, parameter, method), Tensor::doAllocate, &s));
    }

    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
        s |= setResultForBackward(input);
    }
    return services::Status();
}

template DAAL_EXPORT services::Status Input::allocate<DAAL_FPTYPE>(const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace locallyconnected2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
