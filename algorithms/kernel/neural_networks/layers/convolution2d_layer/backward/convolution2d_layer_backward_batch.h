/* file: convolution2d_layer_backward_batch.h */
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
//  Implementation of onvolution2d calculation algorithm and types methods.
//--
*/

#include "convolution2d_layer_backward_types.h"
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
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of backward 2D convolution layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward 2D convolution layer
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input,
                                              const daal::algorithms::Parameter *parameter,
                                              const int method)
{
    using namespace daal::data_management;
    using daal::data_management::TensorPtr;
    using daal::internal::MklTensor;

    const Input *in = static_cast<const Input *>(input);
    const Parameter *param =  static_cast<const Parameter * >(parameter);

    services::Collection<size_t> bDims;
    bDims.push_back(param->nKernels);

    TensorPtr valueTable = in->get(auxData);
    TensorPtr wTable     = in->get(auxWeights);

    if(valueTable == 0 || wTable == 0) return services::Status(services::ErrorNullInputNumericTable);

    if (param->propagateGradient && !get(layers::backward::gradient))
    {
        set(layers::backward::gradient, TensorPtr(
                        new MklTensor<algorithmFPType>(valueTable->getDimensions(), Tensor::doAllocate)));
    }
    if (!get(layers::backward::weightDerivatives))
    {
        set(layers::backward::weightDerivatives, TensorPtr(
                        new MklTensor<algorithmFPType>(wTable->getDimensions(), Tensor::doAllocate)));
    }
    if (!get(layers::backward::biasDerivatives))
    {
        set(layers::backward::biasDerivatives, TensorPtr(
                        new MklTensor<algorithmFPType>(bDims, Tensor::doAllocate)));
    }
    return services::Status();
}

}// namespace interface1
}// namespace forward
}// namespace convolution2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
