/* file: concat_layer_forward_fpt.cpp */
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

#include "concat_layer_forward_types.h"
#include "concat_layer_types.h"
#include "service_mkl_tensor.h"
#include "tensor.h"

using namespace daal::services;
using namespace daal::data_management;

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
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    using daal::data_management::Tensor;
    using daal::data_management::TensorPtr;
    using daal::internal::MklTensor;

    const Input *in = static_cast<const Input * >(input);
    const Parameter *par = static_cast<const Parameter *>(parameter);

    TensorPtr valueTable = in->get(layers::forward::inputLayerData, 0);

    const size_t nInputs = in->get(layers::forward::inputLayerData)->size();
    const size_t concatDimension = par->concatDimension;

    size_t sum = 0;
    for (size_t i = 0; i < nInputs; i++)
    {
        const size_t dim = (in->get(layers::forward::inputLayerData, i))->getDimensionSize(concatDimension);
        sum += dim;
    }
    DAAL_CHECK(valueTable, ErrorNullInputNumericTable);

    Collection<size_t> dimsCollection = valueTable->getDimensions();
    dimsCollection[concatDimension] = sum;

    if (!get(layers::forward::value))
    {
        set(layers::forward::value, TensorPtr(new MklTensor<algorithmFPType>(dimsCollection)));
    }
    set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));

    Status s;
    SharedPtr<data_management::HomogenNumericTable<size_t> > auxDimTable = data_management::HomogenNumericTable<size_t>::create
                                                                                   (nInputs, 1, data_management::NumericTable::doAllocate, &s);

    size_t *auxDimArray = auxDimTable->getArray();

    for (size_t i = 0; i < nInputs; i++)
    {
        size_t dim = (in->get(layers::forward::inputLayerData, i))->getDimensionSize(concatDimension);
        auxDimArray[i] = dim;
    }

    set(layers::concat::auxInputDimensions, auxDimTable);
    return s;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace forward
}// namespace concat
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
