/* file: concat_layer_forward_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of concat calculation algorithm and types methods.
//--
*/

#include "concat_layer_forward_types.h"
#include "concat_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace concat
{
namespace forward
{
namespace interface1
{
/**
* Allocates memory to store the result of the forward concat layer
* \param[in] input     Pointer to an object containing the input data
* \param[in] parameter %Parameter of the algorithm
* \param[in] method    Computation method for the algorithm
*/
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input * >(input);
    const Parameter *par = static_cast<const Parameter *>(parameter);

    services::SharedPtr<data_management::Tensor> valueTable = in->get(layers::forward::inputLayerData, 0);

    size_t nInputs = in->get(layers::forward::inputLayerData)->size();
    size_t concatDimension = par->concatDimension;

    size_t sum = 0;
    for (size_t i = 0; i < nInputs; i++)
    {
        size_t dim = (in->get(layers::forward::inputLayerData, i))->getDimensionSize(concatDimension);
        sum += dim;
    }
    if(!valueTable) { this->_errors->add(services::ErrorNullInputNumericTable); return; }

    services::Collection<size_t> dimsCollection = valueTable->getDimensions();
    dimsCollection[concatDimension] = sum;

    if (!get(layers::forward::value))
    {
        DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::value, dimsCollection);
    }
    set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));

    services::SharedPtr<data_management::HomogenNumericTable<size_t> > auxDimTable(new data_management::HomogenNumericTable<size_t>
                                                                                   (nInputs, 1, data_management::NumericTable::doAllocate));
    size_t *auxDimArray = auxDimTable->getArray();

    for (size_t i = 0; i < nInputs; i++)
    {
        size_t dim = (in->get(layers::forward::inputLayerData, i))->getDimensionSize(concatDimension);
        auxDimArray[i] = dim;
    }

    set(layers::concat::auxInputDimensions, auxDimTable);
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace concat
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
