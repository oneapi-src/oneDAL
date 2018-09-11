/* file: maximum_pooling2d_layer_forward_fpt.cpp */
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
//  Implementation of maximum_pooling2d calculation algorithm and types methods.
//--
*/

#include "maximum_pooling2d_layer_forward_types.h"
#include "maximum_pooling2d_layer_types.h"

#include "service_mkl_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace maximum_pooling2d
{
namespace forward
{
namespace interface1
{
/**
 * Allocates memory to store the result of the forward maximum 2D pooling layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of the forward maximum 2D pooling layer
 * \param[in] method Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::internal::MklTensor;

    services::Status s;
    DAAL_CHECK_STATUS(s, pooling2d::forward::Result::allocate<algorithmFPType>(input, parameter, method));

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    const Input *in = static_cast<const Input *>(input);
    const services::Collection<size_t> &dataDims = in->get(layers::forward::data)->getDimensions();
    services::Collection<size_t> valueDims(dataDims);
    computeValueDimensions(valueDims, algParameter);
    if (get(layers::forward::resultForBackward))
    {
        if (sizeof(size_t) == sizeof(double))
        {
            set(auxSelectedIndices, data_management::TensorPtr(
                    new MklTensor<double>(valueDims, data_management::Tensor::doAllocate)));
        }
        else
        {
            set(auxSelectedIndices, data_management::TensorPtr(
                    new MklTensor<float>(valueDims, data_management::Tensor::doAllocate)));
        }
        if(!algParameter->predictionStage)
        {
            set(auxData, in->get(layers::forward::data));
            set(auxInputDimensions, createAuxInputDimensions(dataDims));
        }
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace maximum_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
