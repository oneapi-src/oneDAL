/* file: transposed_conv2d_layer_backward_batch.h */
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
//  Implementation of transposed convolution2d calculation algorithm and types methods.
//--
*/

#include "transposed_conv2d_layer_backward_types.h"
#include "transposed_conv2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace transposed_conv2d
{
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of backward 2D transposed convolution layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward 2D transposed convolution layer
 * \param[in] method    Computation method
 *
 * \return Status of computations
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::TensorPtr;
    using daal::data_management::HomogenTensor;

    const Input *in = static_cast<const Input *>(input);
    const Parameter *param =  static_cast<const Parameter * >(parameter);

    TensorPtr auxDataTable = in->get(auxData);
    TensorPtr wTable       = in->get(auxWeights);

    services::Collection<size_t> bDims;
    bDims.push_back(param->nKernels);

    DAAL_CHECK(auxDataTable && wTable, services::ErrorNullInputNumericTable);
    services::Status s;
    if (param->propagateGradient && !get(layers::backward::gradient))
    {
        set(layers::backward::gradient, HomogenTensor<algorithmFPType>::create(auxDataTable->getDimensions(), Tensor::doAllocate, &s));
    }
    if (!get(layers::backward::weightDerivatives))
    {
        set(layers::backward::weightDerivatives, HomogenTensor<algorithmFPType>::create(wTable->getDimensions(), Tensor::doAllocate, &s));
    }
    if (!get(layers::backward::biasDerivatives))
    {
        set(layers::backward::biasDerivatives, HomogenTensor<algorithmFPType>::create(bDims, Tensor::doAllocate, &s));
    }
    return s;
}

}// namespace interface1
}// namespace forward
}// namespace transposed_conv2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
