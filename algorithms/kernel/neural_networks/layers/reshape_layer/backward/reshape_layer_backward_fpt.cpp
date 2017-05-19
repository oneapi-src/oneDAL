/* file: reshape_layer_backward_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

        DAAL_ALLOCATE_TENSOR_AND_SET(layers::backward::gradient, iDims);
    }
    return services::Status();
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace reshape
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
