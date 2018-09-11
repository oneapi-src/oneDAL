/* file: reshape_layer_forward_fpt.cpp */
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
