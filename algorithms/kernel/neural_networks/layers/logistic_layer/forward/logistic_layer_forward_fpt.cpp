/* file: logistic_layer_forward_fpt.cpp */
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
//  Implementation of logistic calculation algorithm and types methods.
//--
*/

#include "logistic_layer_forward_types.h"
#include "logistic_layer_types.h"
#include "service_mkl_tensor.h"
#include "service_tensor.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace logistic
{
namespace forward
{
namespace interface1
{

/**
* Allocates memory to store the result of the forward logistic layer
* \param[in] input     Pointer to an object containing the input data
* \param[in] parameter %Parameter of the algorithm
* \param[in] method    Computation method for the algorithm
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);

    services::Status s;
    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        if (!get(layers::forward::value))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, in->get(layers::forward::data)->getDimensions());
        }
        if (!get(layers::forward::resultForBackward))
        {
            set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
        }
        s |= setResultForBackward(input);
    }
    else
    {
        if (!get(layers::forward::value))
        {
            auto inTensor = in->get(layers::forward::data);
            data_management::HomogenTensor<algorithmFPType> *inHomo = dynamic_cast<data_management::HomogenTensor<algorithmFPType>*>( inTensor.get() );
            internal       ::MklTensor    <algorithmFPType> *inMkl  = dynamic_cast<internal       ::MklTensor    <algorithmFPType>*>( inTensor.get() );

            if(inHomo || inMkl)
            {
                set(layers::forward::value, inTensor);
            }
            else
            {
                DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, in->get(layers::forward::data)->getDimensions());
            }
        }
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace logistic
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
