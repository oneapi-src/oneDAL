/* file: eltwise_sum_layer_backward.cpp */
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
//  Implementation of element-wise sum calculation algorithm and types methods.
//--
*/

#include "eltwise_sum_layer_backward_types.h"
#include "eltwise_sum_layer_types.h"
#include "service_numeric_table.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace eltwise_sum
{
namespace backward
{
namespace interface1
{

using namespace daal::data_management;

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_ELTWISE_SUM_BACKWARD_RESULT_ID);

/**
 * Default constructor
 */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input tensor for backward element-twise sum layer
 * \param[in] id Identifier of the input tensor
 * \return       Input tensor that corresponds to the given identifier
 */
TensorPtr Input::get(LayerDataId id) const
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    if (!layerData) { return TensorPtr(); }
    return Tensor::cast((*layerData)[id]);
}

/**
 * Returns an input numeric table for backward element-wise sum layer
 * \param[in] id Identifier of the input numeric table
 * \return       Input numeric table that corresponds to the given identifier
 */
NumericTablePtr Input::get(LayerDataNumericTableId id) const
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    if (!layerData) { return NumericTablePtr(); }
    return NumericTable::cast((*layerData)[id]);
}

/**
 * Sets an input tensor for the backward element-twise sum layer
 * \param[in] id    Identifier of the input tensor
 * \param[in] value Input tensor to set
 */
void Input::set(LayerDataId id, const TensorPtr &value)
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Sets an input numeric table for the backward element-wise sum layer
 * \param[in] id    Identifier of the input numeric table
 * \param[in] value Input numeric table
 */
void Input::set(LayerDataNumericTableId id, const NumericTablePtr &value)
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the element-twise sum layer
 * \param[in] par    %Parameter of layer
 * \param[in] method Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    DAAL_CHECK(layerData, services::ErrorNullLayerData);
    DAAL_CHECK(layerData->size() <= 2, services::ErrorIncorrectSizeOfLayerData);

    services::Status s;
    DAAL_CHECK_STATUS(s, checkInputGradient());
    DAAL_CHECK_STATUS(s, checkAuxCoefficients());
    DAAL_CHECK_STATUS(s, checkAuxNumberOfCoefficients());

    return services::Status();
}

/**
 * Gets number of coefficients (or number of input tensors on the forward pass)
 *
 * \return Number of coefficients
 */
size_t Input::getNumberOfCoefficients() const
{
    TensorPtr auxCoefficients = get(eltwise_sum::auxCoefficients);
    if (auxCoefficients) { return auxCoefficients->getSize(); }

    NumericTablePtr auxCoefficientsNum = get(eltwise_sum::auxNumberOfCoefficients);
    if (auxCoefficientsNum) { return getNumberOfAuxCoefficientsFromTable(); }

    return 0;
}

size_t Input::getNumberOfAuxCoefficientsFromTable() const
{
    NumericTablePtr auxCoefficientsNum = get(eltwise_sum::auxNumberOfCoefficients);
    if (!auxCoefficientsNum) { return 0; }

    BlockDescriptor<int> auxCoefficientsNumBlock;
    auxCoefficientsNum->getBlockOfRows(0, 1, readOnly, auxCoefficientsNumBlock);
    int numberOfAuxCoefficientsValue = *(auxCoefficientsNumBlock.getBlockPtr());
    auxCoefficientsNum->releaseBlockOfRows(auxCoefficientsNumBlock);

    return (size_t)numberOfAuxCoefficientsValue;
}

services::Status Input::checkInputGradient() const
{
    TensorPtr inputGradient = get(layers::backward::inputGradient);
    return data_management::checkTensor(inputGradient.get(), inputGradientStr());
}

services::Status Input::checkAuxCoefficients() const
{
    TensorPtr auxCoefficients = get(eltwise_sum::auxCoefficients);
    if (auxCoefficients)
    {
        services::Status s;
        DAAL_CHECK_STATUS(s, data_management::checkTensor(auxCoefficients.get(), auxCoefficientsStr()));

        DAAL_CHECK_EX(auxCoefficients->getDimensions().size() == 1,
            services::ErrorIncorrectNumberOfDimensionsInTensor, services::ArgumentName, auxCoefficientsStr());
    }
    return services::Status();
}

services::Status Input::checkAuxNumberOfCoefficients() const
{
    TensorPtr auxCoefficients = get(eltwise_sum::auxCoefficients);
    if (!auxCoefficients)
    {
        NumericTablePtr auxNumberOfCoefficients = get(eltwise_sum::auxNumberOfCoefficients);

        const int unexpectedLayouts =
            (int)NumericTableIface::upperPackedSymmetricMatrix  |
            (int)NumericTableIface::lowerPackedSymmetricMatrix  |
            (int)NumericTableIface::upperPackedTriangularMatrix |
            (int)NumericTableIface::lowerPackedTriangularMatrix |
            (int)NumericTableIface::csrArray;

        const size_t requiredNumberOfRows = 1;
        const size_t requiredNumberOfCols = 1;

        services::Status s;
        DAAL_CHECK_NUMERIC_TABLE(s, auxNumberOfCoefficients.get(), auxNumberOfCoefficientsStr(),
            unexpectedLayouts, 0, requiredNumberOfCols, requiredNumberOfRows);

        const size_t numberOfCoefficients = getNumberOfAuxCoefficientsFromTable();
        DAAL_CHECK_EX(numberOfCoefficients > 0, services::ErrorIncorrectInputNumericTable,
                      services::ArgumentName, auxNumberOfCoefficientsStr());
    }
    return services::Status();
}

/**
 * Default constructor
 */
Result::Result() {}

/**
* Returns the result tensor of the backward element-wise layer
* \param[in] id    Identifier of the result tensor
* \param[in] index Index of the result tensor
* \return          Input tensor that corresponds to the given identifier
*/
TensorPtr Result::get(layers::backward::ResultLayerDataId id, size_t index) const
{
    LayerDataPtr layerData = get(id);
    if (!layerData) { return TensorPtr(); }
    return Tensor::cast((*layerData)[index]);
}

/**
 * Sets the result tensor for the backward element-wise layer
 * \param[in] id       Identifier of the result tensor
 * \param[in] value    Pointer to the tensor
 * \param[in] index    Index of the result tensor
 */
void Result::set(layers::backward::ResultLayerDataId id, const TensorPtr &value, size_t index)
{
    LayerDataPtr layerData = get(id);
    (*layerData)[index] = value;
}

/**
 * Returns resulting gradient of the backward element-wise layer
 * \param[in] index Index of the tensor with gradient
 * \return Resulting gradient that corresponds to the given index
 */
TensorPtr Result::getGradient(size_t index) const
{
    return get(layers::backward::resultLayerData, index);
}

/**
 * Returns the layout of the result object for the layer algorithm
 * \return Layout of the result object for the layer algorithm
 */
LayerResultLayout Result::getLayout() const { return collectionResult; }

/**
 * Checks the result of the element-twise sum layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
services::Status Result::check(const daal::algorithms::Input *input,
                               const daal::algorithms::Parameter *par, int method) const
{
    const Input *algInput = dynamic_cast<const Input *>(input);
    DAAL_CHECK(algInput, services::ErrorNullInput);

    services::Status s;
    DAAL_CHECK_STATUS(s, algInput->check(par, method));
    DAAL_CHECK_STATUS(s, checkResultLayerData(algInput));

    return services::Status();
}

services::Status Result::checkResultLayerData(const Input *input) const
{
    LayerDataPtr resultLayerData = get(layers::backward::resultLayerData);
    DAAL_CHECK(resultLayerData, services::ErrorNullLayerData);

    const size_t numberOfCoefficients = input->getNumberOfCoefficients();
    DAAL_CHECK(resultLayerData->size() == numberOfCoefficients, services::ErrorIncorrectSizeOfLayerData);

    return checkOutputGradients(input);
}

services::Status Result::checkOutputGradients(const Input *input) const
{
    LayerDataPtr resultLayerData = get(layers::backward::resultLayerData);

    TensorPtr inputGradient = input->get(layers::backward::inputGradient);
    const services::Collection<size_t> &requiredOutputGradientDims = inputGradient->getDimensions();

    services::Status s;
    for (size_t i = 0; i < resultLayerData->size(); i++)
    {
        TensorPtr outputGradient = Tensor::cast((*resultLayerData)[i]);
        DAAL_CHECK_TENSOR(s, outputGradient.get(), resultLayerDataStr(), &requiredOutputGradientDims);
    }

    return services::Status();
}

void Result::useInputGradientTensorAsOutput(const TensorPtr &inputGradient, size_t nOutputs)
{
    for (size_t i = 0; i < nOutputs; i++)
    {
        set(layers::backward::resultLayerData, inputGradient, i);
    }
}

LayerDataPtr Result::getResultLayerDataAllocateIfEmpty()
{
    LayerDataPtr layerData = get(layers::backward::resultLayerData);
    if (!layerData)
    {
        layerData = LayerDataPtr(new LayerData());
        set(layers::backward::resultLayerData, layerData);
    }
    return layerData;
}

}// namespace interface1
}// namespace backward
}// namespace eltwise_sum
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
