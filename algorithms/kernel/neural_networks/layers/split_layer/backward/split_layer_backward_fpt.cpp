/* file: split_layer_backward_fpt.cpp */
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
//  Implementation of split calculation algorithm and types methods.
//--
*/

#include "split_layer_backward_types.h"
#include "split_layer_types.h"

#include "service_mkl_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace backward
{
namespace interface1
{
/**
* Allocates memory to store the result of the backward split layer
 * \param[in] input     Pointer to an object containing the input data
 * \param[in] method    Computation method for the algorithm
 * \param[in] parameter %Parameter of the backward split layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *param = static_cast<const Parameter *>(parameter);
    if (!param->propagateGradient) { return services::Status(); }

    if (!get(layers::backward::gradient))
    {
        const Input *in = static_cast<const Input *>(input);

        data_management::TensorPtr valueTable = in->get(inputGradientCollection, 0);
        DAAL_CHECK(valueTable, services::ErrorNullInputNumericTable);

        if (!get(layers::backward::gradient))
        {
            set(layers::backward::gradient, data_management::TensorPtr(new internal::MklTensor<algorithmFPType>(
                                                                                        valueTable->getDimensions())));
        }
    }
    return services::Status();
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
