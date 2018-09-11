/* file: fullyconnected_layer_backward_fpt.cpp */
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
//  Implementation of fullyconnected calculation algorithm and types methods.
//--
*/

#include "fullyconnected_layer_backward_types.h"
#include "fullyconnected_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
namespace backward
{
namespace interface1
{
/**
 * Allocates memory to store the result of backward fully-connected layer
 * \param[in] input     Object containing the input data
 * \param[in] parameter %Parameter of backward fully-connected layer
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::TensorPtr;
    using daal::data_management::HomogenTensor;

    const Input *in = static_cast<const Input *>(input);
    const Parameter *param =  static_cast<const Parameter * >(parameter);

    services::Collection<size_t> bDims;
    bDims.push_back(param->nOutputs);

    TensorPtr valueTable = in->get(auxData);
    TensorPtr wTable     = in->get(auxWeights);

    if(!valueTable || !wTable) return services::Status(services::ErrorNullInputNumericTable);
    services::Status s;
    if (param->propagateGradient && !get(layers::backward::gradient))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, valueTable->getDimensions());
    }
    if (!get(layers::backward::weightDerivatives))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::weightDerivatives, wTable->getDimensions());
    }
    if (!get(layers::backward::biasDerivatives))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::biasDerivatives, bDims);
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace fullyconnected
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
