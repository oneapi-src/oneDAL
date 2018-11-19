/* file: locallyconnected2d_layer_backward_batch_fpt.cpp */
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
//  Implementation of locally connected calculation algorithm and types methods.
//--
*/

#include "locallyconnected2d_layer_backward_types.h"
#include "locallyconnected2d_layer_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace locallyconnected2d
{
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of backward 2D locally connected layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward 2D locally connected layer
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
services::Status DAAL_EXPORT Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::HomogenTensor;

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param =  static_cast<const Parameter * >(parameter);

    TensorPtr auxDataTensor = algInput->get(auxData);
    TensorPtr auxWTensor    = algInput->get(auxWeights);

    services::Collection<size_t> bDims;
    getBiasesDims(algInput, param, bDims);

    if(auxDataTensor == 0 || auxWTensor == 0) return services::Status(services::ErrorNullTensor);

    services::Status s;
    if (param->propagateGradient && !get(layers::backward::gradient))
    {
        set(layers::backward::gradient, HomogenTensor<algorithmFPType>::create(auxDataTensor->getDimensions(), Tensor::doAllocate, &s));
    }
    if (!get(layers::backward::weightDerivatives))
    {
        set(layers::backward::weightDerivatives, HomogenTensor<algorithmFPType>::create(auxWTensor->getDimensions(), Tensor::doAllocate, &s));
    }
    if (!get(layers::backward::biasDerivatives))
    {
        set(layers::backward::biasDerivatives, HomogenTensor<algorithmFPType>::create(bDims, Tensor::doAllocate, &s));
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace locallyconnected2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
