/* file: eltwise_sum_layer_forward.cpp */
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
//  Implementation of element-wise sum calculation algorithm and types methods.
//--
*/

#include "eltwise_sum_layer_forward_types.h"
#include "eltwise_sum_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"
#include "homogen_numeric_table.h"

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
namespace forward
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_ELTWISE_SUM_FORWARD_RESULT_ID);

using namespace daal::data_management;

/**
 * Default constructor
 */
Input::Input() : layers::forward::Input(lastInputId + 1) {};
Input::Input(const Input& other) : super(other) {}

/**
* Returns an input tensor of the forward element-wise sum layer
* \param[in] id Identifier of the input tensor
* \return       Input tensor that corresponds to the given identifier
*/
TensorPtr Input::get(InputId id) const
{
    return data_management::Tensor::cast(Argument::get(id));
}

/**
* Sets an input tensor of the forward element-wise sum layer
* \param[in] id    Identifier of the input tensor
* \param[in] value Pointer to the tensor
*/
void Input::set(InputId id, const TensorPtr &value)
{
    Argument::set(id, value);
}

/**
* Returns an input tensor of the forward element-wise sum layer
* \param[in] id    Identifier of the input tensor
* \param[in] index Index of the input tensor
* \return          Input tensor that corresponds to the given identifier
*/
TensorPtr Input::get(layers::forward::InputLayerDataId id, size_t index) const
{
    LayerDataPtr layerData = get(id);
    if (!layerData) { return TensorPtr(); }
    return Tensor::cast((*layerData)[index]);
}

/**
* Sets an input tensor for the forward element-wise sum layer
* \param[in] id    Identifier of the input tensor
* \param[in] value Pointer to the tensor
* \param[in] index Index of the input tensor
*/
void Input::set(layers::forward::InputLayerDataId id, const TensorPtr &value, size_t index)
{
    LayerDataPtr layerData = get(id);
    (*layerData)[index] = value;
}

/**
 * Adds tensor with data to the input object of the forward element-wise sum layer
 * \param[in] dataTensor Tensor with data
 * \param[in] index      Index of the tensor with data
 *
 * \return Status of computations
 */
services::Status Input::addData(const TensorPtr &dataTensor, size_t index)
{
    LayerDataPtr layerData = getInputLayerDataAllocateIfEmpty();
    size_t nInputs = layerData->size();
    (*layerData)[nInputs] = dataTensor;
    return services::Status();
}

/**
 * Erases input data tensor from the input of the forward layer
 *
 * \return Status of computations
 */
services::Status Input::eraseInputData()
{
    set(layers::forward::inputLayerData, LayerDataPtr());
    return services::Status();
}

/**
 * Checks input object of the forward element-wise sum layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    LayerDataPtr layerData = get(layers::forward::inputLayerData);

    DAAL_CHECK(layerData,             services::ErrorNullLayerData);
    DAAL_CHECK(layerData->size() > 0, services::ErrorIncorrectSizeOfLayerData);

    services::Status s;
    DAAL_CHECK_STATUS(s, checkInputTensors(*layerData));
    DAAL_CHECK_STATUS(s, checkCoefficients(*layerData));

    return services::Status();
}

services::Status Input::checkInputTensors(const LayerData &layerData) const
{
    services::Status s;

    TensorPtr firstInput = data_management::Tensor::cast(layerData[0]);
    DAAL_CHECK_STATUS(s, data_management::checkTensor(firstInput.get(), inputLayerDataStr()));

    const services::Collection<size_t> &firstInputDimensions = firstInput->getDimensions();
    for (size_t i = 1; i < layerData.size(); i++)
    {
        TensorPtr input = data_management::Tensor::cast(layerData[i]);
        DAAL_CHECK_TENSOR(s, input.get(), inputLayerDataStr(), &firstInputDimensions);
    }

    return services::Status();
}

services::Status Input::checkCoefficients(const LayerData &layerData) const
{
    TensorPtr coefficients = get(eltwise_sum::forward::coefficients);
    if (coefficients)
    {
        services::Collection<size_t> requiredCoefficientsDimensions;
        requiredCoefficientsDimensions.push_back(layerData.size());

        return data_management::checkTensor(coefficients.get(), coefficientsStr(),
                                            &requiredCoefficientsDimensions);
    }

    return services::Status();
}

LayerDataPtr Input::getInputLayerDataAllocateIfEmpty()
{
    LayerDataPtr layerData = get(layers::forward::inputLayerData);
    if (!layerData)
    {
        layerData = LayerDataPtr(new LayerData());
        set(layers::forward::inputLayerData, layerData);
    }
    return layerData;
}

/**
 * Default constructor
 */
Result::Result()
{
    set(layers::forward::resultForBackward, LayerDataPtr(new LayerData()));
}

/**
* Returns collection of dimensions of element-wise sum layer output
* \param[in] inputSize   Collection of input tensors dimensions
* \param[in] par         Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of element-wise sum layer output
*/
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    return services::Collection<size_t>();
}

/**
* Returns collection of dimensions of element-wise sum layer output
* \param[in] inputSize   Collection of input tensors dimensions
* \param[in] parameter   Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of element-wise sum layer output
*/
services::Collection<size_t> Result::getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                                  const daal::algorithms::Parameter *parameter, const int method)
{
    if (inputSize.size() > 0) { return inputSize[0]; }
    return services::Collection<size_t>();
}

/**
 * Returns the result tensor of forward element-wise sum layer
 * \param[in] id Identifier of the result tensor
 * \return       Result tensor that corresponds to the given identifier
 */
TensorPtr Result::get(LayerDataId id) const
{
    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    if (!layerData) { return TensorPtr(); }
    return data_management::Tensor::cast((*layerData)[id]);
}

/**
 * Returns the result numeric table of the forward element-wise sum layer
 * \param[in] id Identifier of the result numeric table
 * \return       Result numeric table that corresponds to the given identifier
 */
NumericTablePtr Result::get(LayerDataNumericTableId id) const
{
    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    if (!layerData) { return NumericTablePtr(); }
    return data_management::NumericTable::cast((*layerData)[id]);
}

/**
 * Sets the result tensor of forward element-wise sum layer
 * \param[in] id    Identifier of the result tensor
 * \param[in] value Result tensor
 */
void Result::set(LayerDataId id, const TensorPtr &value)
{
    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    (*layerData)[id] = value;
}

/**
 * Sets the result numeric table of the forward element-wise sum layer
 * \param[in] id  Identifier of the result numeric table
 * \param[in] ptr Result numeric tensor
 */
void Result::set(LayerDataNumericTableId id, const NumericTablePtr &value)
{
    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    (*layerData)[id] = value;
}

/**
 * Sets the result that is used in backward layer
 * \param[in] input  Pointer to an object containing the input data
 *
 * \return Status of operation
 */
services::Status Result::setResultForBackward(const daal::algorithms::Input *input)
{
    services::Status s;
    const Input *eltwiseInput = dynamic_cast<const Input *>(input);
    DAAL_CHECK(eltwiseInput, services::ErrorNullInput);

    TensorPtr coefficients = eltwiseInput->get(eltwise_sum::forward::coefficients);
    if (coefficients)
    {
        // We do not copy input coefficients to auxCoefficients
        // and just save the pointer to input coefficients
        set(eltwise_sum::auxCoefficients, coefficients);
    }
    else
    {
        size_t numberOfCoefficientsTableValue =
            eltwiseInput->get(layers::forward::inputLayerData)->size();

        NumericTablePtr numberOfCoefficients = HomogenNumericTable<int>::create(
            1, 1, NumericTable::doAllocate, (int)numberOfCoefficientsTableValue, &s);
        DAAL_CHECK_STATUS_VAR(s);

        set(eltwise_sum::auxNumberOfCoefficients, numberOfCoefficients);
    }

    return services::Status();
}

/**
 * Checks the result of the forward element-wise sum layer
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
    DAAL_CHECK_STATUS(s, checkValue(algInput));
    DAAL_CHECK_STATUS(s, checkAuxCoefficients(algInput));
    DAAL_CHECK_STATUS(s, checkAuxNumberOfCoefficients(algInput));

    return services::Status();
}

services::Status Result::checkValue(const Input *input) const
{
    TensorPtr firstInput = input->get(layers::forward::inputLayerData, 0);
    const services::Collection<size_t> &firstInputDimensions = firstInput->getDimensions();

    TensorPtr value = get(layers::forward::value);
    return data_management::checkTensor(value.get(), valueStr(), &firstInputDimensions);
}

services::Status Result::checkAuxCoefficients(const Input *input) const
{
    TensorPtr coefficients = input->get(eltwise_sum::forward::coefficients);
    if (coefficients)
    {
        TensorPtr auxCoefficients = get(eltwise_sum::auxCoefficients);
        const services::Collection<size_t> &requiredAuxCoefficientsDimensions = coefficients->getDimensions();

        return data_management::checkTensor(auxCoefficients.get(), auxCoefficientsStr(),
                                            &requiredAuxCoefficientsDimensions);
    }

    return services::Status();
}

services::Status Result::checkAuxNumberOfCoefficients(const Input *input) const
{
    TensorPtr coefficients = input->get(eltwise_sum::forward::coefficients);
    if (!coefficients)
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

        return data_management::checkNumericTable(auxNumberOfCoefficients.get(), auxNumberOfCoefficientsStr(),
                                                  unexpectedLayouts, 0, requiredNumberOfCols, requiredNumberOfRows);
    }

    return services::Status();
}

LayerDataPtr Result::getResultLayerDataAllocateIfEmpty()
{
    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    if (!layerData)
    {
        layerData = LayerDataPtr(new LayerData());
        set(layers::forward::resultForBackward, layerData);
    }
    return layerData;
}

} // namespace interface1
} // namespace forward
} // namespace eltwise_sum
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
