/* file: eltwise_sum_layer_backward_fpt.cpp */
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
//  Implementation of element-wise sum calculation algorithm and types methods.
//--
*/

#include "eltwise_sum_layer_backward_types.h"
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
namespace backward
{
namespace interface1
{

using namespace daal::data_management;

/**
 * Allocates memory to store the result of backward element-wise sum layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward element-wise sum layer
 * \param[in] method    Computation method
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

    const size_t nOutputs     = eltwiseInput->getNumberOfCoefficients();
    TensorPtr auxCoefficients = eltwiseInput->get(eltwise_sum::auxCoefficients);
    TensorPtr inputGradient   = eltwiseInput->get(layers::backward::inputGradient);

    // We call this function just to allocate
    // resultLayerData and consciously ignore returned value
    LayerDataPtr resultLayerData = getResultLayerDataAllocateIfEmpty();
    DAAL_CHECK_MALLOC(resultLayerData);

    return allocateNewOutputTensors<algorithmFPType>(inputGradient, nOutputs);
}

template <typename algorithmFPType>
services::Status Result::allocateNewOutputTensors(const TensorPtr &inputGradient, size_t nOutputs)
{
    const services::Collection<size_t> &inputGradientDimensions = inputGradient->getDimensions();
    services::Status s;
    for (size_t i = 0; i < nOutputs; i++)
    {
        TensorPtr resultLayerDataI = HomogenTensor<algorithmFPType>::create(inputGradientDimensions, Tensor::doAllocate, &s);
        DAAL_CHECK_STATUS_VAR(s);

        set(layers::backward::resultLayerData, resultLayerDataI, i);
    }

    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(
    const daal::algorithms::Input *, const daal::algorithms::Parameter *, const int);

template DAAL_EXPORT services::Status Result::allocateNewOutputTensors<DAAL_FPTYPE>(const TensorPtr &, size_t);

}// namespace interface1
}// namespace backward
}// namespace eltwise_sum
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
