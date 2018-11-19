/* file: lcn_layer_backward_batch.h */
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

#include "lcn_layer_backward_types.h"
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
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of backward local contrast normalization layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward local contrast normalization layer
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::TensorPtr;
    using daal::data_management::HomogenTensor;

    const Parameter *param =  static_cast<const Parameter * >(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    const Input *in = static_cast<const Input *>(input);

    TensorPtr centeredDataTensor = in->get(auxCenteredData);
    TensorPtr sigmaTensor        = in->get(auxSigma);
    TensorPtr cTensor            = in->get(auxC);
    TensorPtr maxTensor          = in->get(auxInvMax);

    DAAL_CHECK(centeredDataTensor != 0 && sigmaTensor != 0 && cTensor != 0 && maxTensor != 0, services::ErrorNullTensor);
    services::Status s;
    if (!get(layers::backward::gradient))
    {
        set(layers::backward::gradient, HomogenTensor<algorithmFPType>::create(centeredDataTensor->getDimensions(), Tensor::doAllocate, &s));
    }
    return s;
}

}// namespace interface1
}// namespace forward
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
