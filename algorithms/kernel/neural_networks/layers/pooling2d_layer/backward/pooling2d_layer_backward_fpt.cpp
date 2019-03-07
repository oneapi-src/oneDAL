/* file: pooling2d_layer_backward_fpt.cpp */
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
//  Implementation of pooling2d calculation algorithm and types methods.
//--
*/

#include "pooling2d_layer_backward_types.h"
#include "pooling2d_layer_types.h"
#include "daal_strings.h"

#include "service_mkl_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace pooling2d
{
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of the backward 2D pooling layer
 * \param[in] input Pointer to an object containing the input data
 * \param[in] method Computation method for the layer
 * \param[in] parameter %Parameter of the backward 2D pooling layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::internal::MklTensor;

    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    const Input *in = static_cast<const Input *>(input);

    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(in->get(layers::backward::inputGradient).get(), inputGradientStr()));

    if (!get(layers::backward::gradient))
    {
        set(layers::backward::gradient, data_management::TensorPtr(
                new MklTensor<algorithmFPType>(in->getGradientSize(), data_management::Tensor::doAllocate)));
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
