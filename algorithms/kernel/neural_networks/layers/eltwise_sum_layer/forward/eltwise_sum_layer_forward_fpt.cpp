/* file: eltwise_sum_layer_forward_fpt.cpp */
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
//  Implementation of element-wise sum calculation algorithm and types methods.
//--
*/

#include "eltwise_sum_layer_forward_types.h"
#include "eltwise_sum_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
namespace forward
{
namespace interface1
{

using namespace daal::data_management;

/**
 * Allocates memory to store the result of forward  element-wise sum layer
 * \param[in] input     %Input object for the algorithm
 * \param[in] parameter %Parameter of forward element-wise sum layer
 * \param[in] method    Computation method for the layer
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input,
                                              const daal::algorithms::Parameter *parameter,
                                              const int method)
{
    services::Status s;
    const Input *eltwiseInput = dynamic_cast<const Input *>(input);
    DAAL_CHECK(eltwiseInput, services::ErrorNullInput);
    DAAL_CHECK_STATUS(s, eltwiseInput->check(parameter, method));
    DAAL_CHECK_STATUS(s, allocateValueTensor<algorithmFPType>(eltwiseInput));

    // We call this function just to allocate
    // resultLayerData and consciously ignore returned value
    LayerDataPtr resultLayerData = getResultLayerDataAllocateIfEmpty();
    DAAL_CHECK_MALLOC(resultLayerData);

    return setResultForBackward(input);
}

template<typename algorithmFPType>
services::Status Result::allocateValueTensor(const Input *eltwiseInput)
{
    TensorPtr firstInput = eltwiseInput->get(layers::forward::inputLayerData, 0);
    services::Status s;
    TensorPtr value = HomogenTensor<algorithmFPType>::create(firstInput->getDimensions(), Tensor::doAllocate, &s);
    DAAL_CHECK_MALLOC(value);

    set(layers::forward::value, value);

    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(
    const daal::algorithms::Input *, const daal::algorithms::Parameter *, const int);

template DAAL_EXPORT services::Status Result::allocateValueTensor<DAAL_FPTYPE>(const Input *);

}// namespace interface1
}// namespace forward
}// namespace eltwise_sum
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
