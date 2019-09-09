/* file: concat_layer_backward_fpt.cpp */
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
//  Implementation of concat calculation algorithm and types methods.
//--
*/

#include "concat_layer_backward_types.h"
#include "concat_layer_types.h"

#include "service_mkl_tensor.h"

using namespace daal::data_management;
using namespace daal::services;

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
namespace backward
{
namespace interface1
{
/**
* Allocates memory to store the result of the backward concat layer
 * \param[in] input     Pointer to an object containing the input data
 * \param[in] method    Computation method for the algorithm
 * \param[in] parameter %Parameter of the backward concat layer
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *par = static_cast<const Parameter *>(parameter);
    if (!par->propagateGradient) { return Status(); }

    LayerDataPtr layerData = get(layers::backward::resultLayerData);
    if (layerData && layerData->size() > 0) { return Status(); }

    const Input *in = static_cast<const Input * >(input);

    const size_t concatDimension = par->concatDimension;

    const size_t nOutputs = (in->get(layers::concat::auxInputDimensions))->getNumberOfColumns();

    LayerDataPtr resultCollection = LayerDataPtr(new LayerData());

    Collection<size_t> dimsCollection = in->get(layers::backward::inputGradient)->getDimensions();

    for(size_t i = 0; i < nOutputs; i++)
    {
        NumericTablePtr dimsTable = in->get(layers::concat::auxInputDimensions);

        dimsCollection[concatDimension] = getElem(dimsTable, i);
        (*resultCollection)[i] = TensorPtr(new internal::MklTensor<algorithmFPType>(
                                                                                  dimsCollection, Tensor::doAllocate));
    }
    set(layers::backward::resultLayerData, resultCollection);
    return Status();
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace backward
}// namespace concat
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
