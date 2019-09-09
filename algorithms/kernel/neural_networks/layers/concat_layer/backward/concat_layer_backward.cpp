/* file: concat_layer_backward.cpp */
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
#include "serialization_utils.h"
#include "daal_strings.h"
#include "service_numeric_table.h"

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
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONCAT_BACKWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
* Returns input object of the backward concat layer
* \param[in] id    Identifier of the input object
* \return          %Input LayerData that corresponds to the given identifier
*/
NumericTablePtr Input::get(layers::concat::LayerDataId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>
           ((*get(layers::backward::inputFromForward))[id]);
}

/**
* Sets input for the backward concat layer
* \param[in] id      Identifier of the input object
* \param[in] value   Pointer to the object
*/
void Input::set(layers::concat::LayerDataId id, const NumericTablePtr &value)
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
* Checks an input object for the layer algorithm
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method of the algorithm
*/
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(par);
    if (!algParameter->propagateGradient) { return Status(); }

    const size_t concatDimension = algParameter->concatDimension;

    Status s;
    TensorPtr inputGradientTensor = get(layers::backward::inputGradient);
    DAAL_CHECK_STATUS(s, checkTensor(inputGradientTensor.get(), inputGradientStr()));
    Collection<size_t> inputGradientDims = inputGradientTensor->getDimensions();
    DAAL_CHECK((concatDimension <= inputGradientDims.size() - 1), ErrorIncorrectParameter);
    DAAL_CHECK((Argument::size() == 2), ErrorIncorrectNumberOfInputNumericTables);

    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    DAAL_CHECK(layerData, ErrorNullLayerData);


    NumericTablePtr dimsNT = get(auxInputDimensions);
    DAAL_CHECK((dimsNT.get() != 0), ErrorNullNumericTable);
    DAAL_CHECK((dimsNT->getNumberOfRows() == 1), ErrorIncorrectNumberOfRowsInInputNumericTable);
    DAAL_CHECK((dimsNT->getNumberOfColumns() != 0), ErrorIncorrectNumberOfColumnsInInputNumericTable);

    const size_t inputSize = dimsNT->getNumberOfColumns();

    data_management::BlockDescriptor<int> block;
    dimsNT->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *auxDims = block.getBlockPtr();

    size_t sum = 0;
    for (size_t i = 0; i < inputSize; i++)
    {
        sum += auxDims[i];
    }

    DAAL_CHECK((inputGradientDims[concatDimension] == sum), ErrorIncorrectNumberOfRowsInInputNumericTable);
    inputGradientDims[concatDimension] = sum;

    return checkTensor(inputGradientTensor.get(), inputGradientStr(), &inputGradientDims);
}

    /** \brief Default constructor */
Result::Result() : layers::backward::Result() {};

/**
* Returns result object of the backward concat layer
* \param[in] id       Identifier of the result object
* \param[in] index    Index of the result object
* \return             %Input ResultLayerData that corresponds to the given identifier
*/
TensorPtr Result::get(layers::backward::ResultLayerDataId id, size_t index) const
{
    LayerDataPtr layerData = get(id);
    return staticPointerCast<Tensor, SerializationIface>((*layerData)[index]);
}

/**
 * Sets result for the backward concat layer
 * \param[in] id       Identifier of the result object
 * \param[in] value    Pointer to the object
 * \param[in] index    Index of the result object
 */
void Result::set(layers::backward::ResultLayerDataId id, const TensorPtr &value, size_t index)
{
    LayerDataPtr layerData = get(id);
    (*layerData)[index] = value;
}

/**
 * Returns resulting gradient of the backward concat layer
 * \param[in] index Index of the tensor with gradient
 * \return Resulting gradient that corresponds to the given index
 */
TensorPtr Result::getGradient(size_t index) const
{
    return get(layers::backward::resultLayerData, index);
}

/**
 * Checks the result of the backward concat layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return Status(); }

    DAAL_CHECK((Argument::size() == 4), ErrorIncorrectNumberOfInputNumericTables);

    const Input *algInput = static_cast<const Input *>(input);
    const size_t concatDimension = parameter->concatDimension;
    LayerDataPtr layerData = get(layers::backward::resultLayerData);
    DAAL_CHECK(layerData, ErrorNullLayerData);

    const size_t nInputs = layerData->size();
    DAAL_CHECK((nInputs != 0), ErrorIncorrectSizeOfLayerData);

    TensorPtr inputGradientTensor = algInput->get(layers::backward::inputGradient);

    Collection<size_t> dims = inputGradientTensor->getDimensions();

    Status s;
    size_t sum = 0;
    for (size_t i = 0; i < nInputs; i++)
    {
        TensorPtr layerDataTensor = get(layers::backward::resultLayerData, i);
        dims[concatDimension] = layerDataTensor->getDimensionSize(concatDimension);
        sum += dims[concatDimension];

        DAAL_CHECK_STATUS(s, checkTensor(layerDataTensor.get(), resultLayerDataStr(), &dims));
    }
    DAAL_CHECK((sum == inputGradientTensor->getDimensionSize(concatDimension)), Error::create(ErrorIncorrectSizeOfDimensionInTensor, ArgumentName, inputGradientStr()));

    return s;
}

/**
 * Returns the layout of the result object for the layer algorithm
 * \return Layout of the result object for the layer algorithm
 */
LayerResultLayout Result::getLayout() const  { return collectionResult; }

size_t Result::getElem(NumericTablePtr nt, size_t index) const
{
    BlockDescriptor<int> block;
    nt->getBlockOfRows(0, 1, readOnly, block);
    int *dataArray = block.getBlockPtr();
    nt->releaseBlockOfRows(block);
    return (size_t)dataArray[index];
}

}// namespace interface1
}// namespace backward
}// namespace concat
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
