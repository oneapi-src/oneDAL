/* file: elu_layer_backward_fpt.cpp */
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
//  Implementation of ELU calculation algorithm and types methods.
//--
*/

#include "elu_layer_backward_types.h"
#include "elu_layer_types.h"

#include "daal_strings.h"
#include "service_mkl_tensor.h"

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
namespace backward
{
namespace interface1
{

using namespace daal::services;
using namespace daal::data_management;

/**
* Allocates memory to store the result of the backward ELU layer
 * \param[in] input     Pointer to an object containing the input data
 * \param[in] method    Computation method for the algorithm
 * \param[in] parameter %Parameter of the backward ELU layer
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input,
                                    const daal::algorithms::Parameter *parameter,
                                    const int method)
{
    Status status;

    auto *in  = static_cast<const Input *>(input);
    auto *par = static_cast<const Parameter *>(parameter);
    if (!par->propagateGradient) { return services::Status(); }

    const Tensor *inputGradientTensor = in->get(layers::backward::inputGradient).get();
    DAAL_CHECK(inputGradientTensor, Error::create(ErrorNullTensor, ArgumentName, inputGradientStr()));

    if (!get(layers::backward::gradient))
    {
        using daal::internal::createTensorKeepingType;
        const TensorPtr gradientTensor = createTensorKeepingType<algorithmFPType>(inputGradientTensor, status);

        DAAL_CHECK_STATUS_VAR(status);
        set(layers::backward::gradient, gradientTensor);
    }
    return services::Status();
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input,
                                                          const daal::algorithms::Parameter *parameter,
                                                          const int method);

} // namespace interface1
} // namespace backward
} // namespace elu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
