/* file: concat_layer_backward_fpt.cpp */
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
