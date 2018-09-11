/* file: convolution2d_layer_forward_batch_fpt.cpp */
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
//  Implementation of onvolution2d calculation algorithm and types methods.
//--
*/

#include "convolution2d_layer_forward_types.h"
#include "convolution2d_layer_types.h"

#include "service_mkl_tensor.h"

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
namespace forward
{
namespace interface1
{

/**
* Allocates memory to store the result of forward  2D convolution layer
 * \param[in] parameter %Parameter of forward 2D convolution layer
 * \param[in] method    Computation method for the layer
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Input::allocate(const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::services::SharedPtr;
    using daal::data_management::Tensor;
    using daal::internal::MklTensor;

    const Parameter *param =  static_cast<const Parameter * >(parameter);

    if( !get(layers::forward::weights) )
    {
        SharedPtr<Tensor> tensor(new MklTensor<algorithmFPType>(getWeightsSizes(param), Tensor::doAllocate));
        set(layers::forward::weights, tensor);
    }

    if( !get(layers::forward::biases) )
    {
        SharedPtr<Tensor> tensor(new MklTensor<algorithmFPType>(getBiasesSizes(param), Tensor::doAllocate));
        set(layers::forward::biases, tensor);
    }
    return services::Status();
}

/**
 * Allocates memory to store the result of forward  2D convolution layer
 * \param[in] input     %Input object for the algorithm
 * \param[in] parameter %Parameter of forward 2D convolution layer
 * \param[in] method    Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::internal::MklTensor;
    const Input *in = static_cast<const Input * >(input);

    const services::Collection<size_t> &inDims = in->get(layers::forward::data)->getDimensions();

    if (!get(layers::forward::value))
    {
        set(layers::forward::value, services::SharedPtr<Tensor>(
                new MklTensor<algorithmFPType>(getValueSize(inDims, parameter, method), Tensor::doAllocate)));
    }
    services::Status s;
    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
        s |= setResultForBackward(input);
    }
    return s;
}

template DAAL_EXPORT services::Status Input::allocate<DAAL_FPTYPE>(const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace convolution2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
