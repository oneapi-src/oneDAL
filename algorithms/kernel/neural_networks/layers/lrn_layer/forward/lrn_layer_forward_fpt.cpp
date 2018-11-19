/* file: lrn_layer_forward_fpt.cpp */
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
//  Implementation of lrn calculation algorithm and types methods.
//--
*/

#include "lrn_layer_forward_types.h"
#include "lrn_layer_types.h"

#include "tensor.h"
#include "service_mkl_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lrn
{
namespace forward
{
namespace interface1
{

using namespace daal::data_management;

/**
 * Allocates memory to store the result of the forward local response normalization layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the layer
 * \param[in] parameter %Parameter of the forward local response normalization layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::TensorPtr;
    using daal::internal::MklTensor;

    const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);

    if (!get(layers::forward::value))
    {
        set(layers::forward::value, TensorPtr(new MklTensor<algorithmFPType>(in->get(layers::forward::data)->getDimensions())));
    }
    if (!get(layers::forward::resultForBackward))
    {
        set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
    }
    if (!get(lrn::auxSmBeta))
    {
        set(lrn::auxSmBeta, TensorPtr(new MklTensor<algorithmFPType>(in->get(layers::forward::data)->getDimensions())));
    }

    services::Status s;
    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        s |= setResultForBackward(input);
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace lrn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
