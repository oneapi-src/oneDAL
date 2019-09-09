/* file: reshape_layer_forward_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "reshape_layer_forward_types.h"
#include "reshape_layer_types.h"
#include "service_numeric_table.h"

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
namespace forward
{
namespace interface1
{
/**
* Allocates memory to store the result of the forward reshape layer
* \param[in] input     Pointer to an object containing the input data
* \param[in] parameter %Parameter of the layer
* \param[in] method    Computation method
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const layers::forward::Input *in = static_cast<const layers::forward::Input * >(input);
    const layers::reshape::Parameter *par = static_cast<const layers::reshape::Parameter * >(parameter);

    set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
    services::Status s;
    if (!get(layers::forward::value))
    {
        const services::Collection<size_t> inDims = in->get(layers::forward::data)->getDimensions();
        services::Collection<size_t> outDims = par->reshapeDimensions;

        bool haveNegative = false;
        size_t negIndex = 0;
        size_t nonNegSize = 1;

        for( size_t i = 0; i < outDims.size(); i++ )
        {
            if( outDims[i] == undefinedDimensionSize )
            {
                haveNegative = true;
                negIndex = i;
            }
            else
            {
                if( outDims[i] == 0 )
                {
                    outDims[i] = inDims[i];
                }

                nonNegSize *= outDims[i];
            }
        }

        if(haveNegative)
        {
            outDims[negIndex] = in->get(layers::forward::data)->getSize() / nonNegSize;
        }

        DAAL_ALLOCATE_TENSOR_AND_SET(s, layers::forward::value, outDims);

        services::SharedPtr<data_management::HomogenNumericTable<size_t> >
        auxDimTable = data_management::HomogenNumericTable<size_t>::create(inDims.size(), 1, data_management::NumericTable::doAllocate, &s);

        size_t *auxDimArray = auxDimTable->getArray();

        for( size_t i = 0; i < inDims.size(); i++ )
        {
            auxDimArray[i] = inDims[i];
        }

        set(layers::reshape::auxInputDimensions, auxDimTable);
    }
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace reshape
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
