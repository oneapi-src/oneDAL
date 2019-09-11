/* file: reshape_layer_backward_fpt.cpp */
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
//  Implementation of reshape calculation algorithm and types methods.
//--
*/

#include "reshape_layer_backward_types.h"
#include "reshape_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace reshape
{
namespace backward
{
namespace interface1
{
/**
* Allocates memory to store the result of the backward reshape layer
* \param[in] input     Pointer to an object containing the input data
* \param[in] parameter %Parameter of the layer
* \param[in] method    Computation method
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const layers::Parameter *param = static_cast<const layers::Parameter * >(parameter);
    if (!param->propagateGradient) { return services::Status(); }
    const Input *in = static_cast<const Input *>(input);
    services::Status s;
    if (!get(layers::backward::gradient))
    {
        data_management::NumericTablePtr dimsTable = in->get(layers::reshape::auxInputDimensions);

        size_t nDims = dimsTable->getNumberOfColumns();

        services::Collection<size_t> iDims( nDims );

        data_management::BlockDescriptor<int> block;
        dimsTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *dataArray = block.getBlockPtr();

        for(size_t i = 0; i < nDims; i++)
        {
            iDims[i] = dataArray[i];
        }

        dimsTable->releaseBlockOfRows(block);

        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::backward::gradient, iDims);
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace reshape
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
