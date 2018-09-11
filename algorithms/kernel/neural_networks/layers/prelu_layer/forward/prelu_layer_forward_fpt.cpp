/* file: prelu_layer_forward_fpt.cpp */
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
//  Implementation of prelu calculation algorithm and types methods.
//--
*/

#include "prelu_layer_forward_types.h"
#include "prelu_layer_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace prelu
{
namespace forward
{
namespace interface1
{
/**
* Allocates memory to store weights for the forward prelu layer
* \param[in] par      %Parameter of the algorithm
* \param[in] method   Computation method for the algorithm
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Input::allocate(const daal::algorithms::Parameter *par, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::TensorPtr;
    using daal::data_management::HomogenTensor;

    const Parameter *parameter = static_cast<const Parameter * >(par);
    TensorPtr dataTensor = get(layers::forward::data);
    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(dataTensor.get(), dataStr()))

    size_t wStartDim = parameter->dataDimension;
    size_t wDimNumber = parameter->weightsDimension;

    if(wStartDim > dataTensor->getNumberOfDimensions() - 1 || wStartDim + wDimNumber > dataTensor->getNumberOfDimensions())
    { return services::Status(services::ErrorIncorrectParameter); }

    if( get(layers::forward::weights) == 0 )
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::weights, getWeightsSizes(parameter));
    }
    return s;
}

/**
* Allocates memory to store the result of the forward prelu layer
* \param[in] input    Pointer to an object containing the input data
* \param[in] par      %Parameter of the algorithm
* \param[in] method   Computation method for the algorithm
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    const Input *in = static_cast<const Input *>(input);
    services::Status s;
    if (!get(layers::forward::value))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, in->get(layers::forward::data)->getDimensions());
    }
    const layers::Parameter *parameter = static_cast<const layers::Parameter * >(par);
    if(!parameter->predictionStage)
    {
        if (!get(layers::forward::resultForBackward))
        {
            set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
        }
        setResultForBackward(input);
    }
    return s;
}

template DAAL_EXPORT services::Status Input::allocate<DAAL_FPTYPE>(const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace prelu
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
