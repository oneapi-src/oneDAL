/* file: elu_layer_forward_fpt.cpp */
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
//  Implementation of ELU calculation algorithm and types methods.
//--
*/

#include "elu_layer_forward_types.h"
#include "elu_layer_types.h"
#include "service_mkl_tensor.h"
#include "tensor.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace elu
{
namespace forward
{
namespace interface1
{

using namespace daal::services;
using namespace daal::data_management;

/**
* Allocates memory to store the result of the forward ELU layer
* \param[in] input      Pointer to an object containing the input data
* \param[in] parameter  %Parameter of the algorithm
* \param[in] method     Computation method for the algorithm
*/
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input,
                                    const daal::algorithms::Parameter *parameter,
                                    const int method)
{
    Status status;

    auto in  = static_cast<const layers::forward::Input *>(input);
    auto par = static_cast<const layers::Parameter *>(parameter);

    const Tensor *dataTensor = in->get(layers::forward::data).get();
    DAAL_CHECK(dataTensor, Error::create(ErrorNullTensor, ArgumentName, dataStr()));

    if (!get(layers::forward::value))
    {
        using daal::internal::createTensorKeepingType;
        const TensorPtr valueTensor = createTensorKeepingType<algorithmFPType>(dataTensor, status);

        DAAL_CHECK_STATUS_VAR(status);
        set(layers::forward::value, valueTensor);
    }

    if (!get(layers::forward::resultForBackward) && !par->predictionStage)
    {
        LayerDataPtr layerData(new LayerData());

        DAAL_CHECK(layerData, services::ErrorMemoryAllocationFailed);
        set(layers::forward::resultForBackward, layerData);
    }

    if (!get(layers::elu::auxIntermediateValue) && !par->predictionStage)
    {
        const TensorPtr auxIntermediateValueTensor = HomogenTensor<algorithmFPType>::create(
            dataTensor->getDimensions(), Tensor::doAllocate, &status);

        DAAL_CHECK_STATUS_VAR(status);
        set(layers::elu::auxIntermediateValue, auxIntermediateValueTensor);
    }

    if (!par->predictionStage)
    {
        DAAL_CHECK_STATUS(status, setResultForBackward(input));
    }

    return status;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                          const daal::algorithms::Parameter *parameter,
                                                          const int method);

} // namespace interface1
} // namespace forward
} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
