/* file: concat_layer_backward.cpp */
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

#include "concat_layer_backward_types.h"
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
namespace backward
{
namespace interface1
{
/** \brief Default constructor */
Input::Input() {};

/**
* Returns input object of the backward concat layer
* \param[in] id    Identifier of the input object
* \return          %Input LayerData that corresponds to the given identifier
*/
data_management::NumericTablePtr Input::get(layers::concat::LayerDataId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>
           ((*get(layers::backward::inputFromForward))[id]);
}

/**
* Sets input for the backward concat layer
* \param[in] id      Identifier of the input object
* \param[in] value   Pointer to the object
*/
void Input::set(layers::concat::LayerDataId id, const data_management::NumericTablePtr &value)
{
    services::SharedPtr<LayerData> layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
* Checks an input object for the layer algorithm
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method of the algorithm
*/
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(par);
    size_t concatDimension = algParameter->concatDimension;

    services::SharedPtr<data_management::Tensor> inputGradientTensor = get(layers::backward::inputGradient);
    if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr())) { return; }
    services::Collection<size_t> inputGradientDims = inputGradientTensor->getDimensions();
    if (concatDimension > inputGradientDims.size() - 1) {this->_errors->add(services::ErrorIncorrectParameter); return; }

    if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    services::SharedPtr<LayerData> layerData = get(layers::backward::inputFromForward);
    if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }


    data_management::NumericTablePtr dimsNT = get(auxInputDimensions);
    if (dimsNT->getNumberOfRows() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
    if (dimsNT->getNumberOfColumns() == 0) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInInputNumericTable); return; }

    size_t inputSize = dimsNT->getNumberOfColumns();

    data_management::BlockDescriptor<int> block;
    dimsNT->getBlockOfRows(0, 1, data_management::readOnly, block);
    int *auxDims = block.getBlockPtr();

    size_t sum = 0;
    for (size_t i = 0; i < inputSize; i++)
    {
        sum += auxDims[i];
    }

    if (inputGradientDims[concatDimension] != sum) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
    inputGradientDims[concatDimension] = sum;

    if (!data_management::checkTensor(inputGradientTensor.get(), this->_errors.get(), inputGradientStr(), &inputGradientDims)) { return; }

    dimsNT->releaseBlockOfRows(block);
}

    /** \brief Default constructor */
Result::Result() : layers::backward::Result() {};

/**
* Returns result object of the backward concat layer
* \param[in] id       Identifier of the result object
* \param[in] index    Index of the result object
* \return             %Input ResultLayerData that corresponds to the given identifier
*/
services::SharedPtr<data_management::Tensor> Result::get(layers::backward::ResultLayerDataId id, size_t index) const
{
    services::SharedPtr<LayerData> layerData = get(id);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[index]);
}

/**
 * Sets result for the backward concat layer
 * \param[in] id       Identifier of the result object
 * \param[in] value    Pointer to the object
 * \param[in] index    Index of the result object
 */
void Result::set(layers::backward::ResultLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
{
    services::SharedPtr<LayerData> layerData = get(id);
    (*layerData)[index] = value;
}

/**
 * Returns resulting gradient of the backward concat layer
 * \param[in] index Index of the tensor with gradient
 * \return Resulting gradient that corresponds to the given index
 */
services::SharedPtr<data_management::Tensor> Result::getGradient(size_t index) const
{
    return get(layers::backward::resultLayerData, index);
}

/**
 * Checks the result of the backward concat layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 4) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *parameter = static_cast<const Parameter *>(par);
    size_t concatDimension = parameter->concatDimension;
    services::SharedPtr<LayerData> layerData = get(layers::backward::resultLayerData);
    if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

    size_t nInputs = layerData->size();
    if (nInputs == 0) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }

    services::SharedPtr<data_management::Tensor> inputGradientTensor = algInput->get(layers::backward::inputGradient);

    services::Collection<size_t> dims = inputGradientTensor->getDimensions();

    size_t sum = 0;
    for (size_t i = 0; i < nInputs; i++)
    {
        services::SharedPtr<data_management::Tensor> layerDataTensor = get(layers::backward::resultLayerData, i);
        dims[concatDimension] = layerDataTensor->getDimensionSize(concatDimension);
        sum += dims[concatDimension];

        if (!data_management::checkTensor(layerDataTensor.get(), this->_errors.get(), resultLayerDataStr(), &dims)) { return; }
    }
    if (sum != inputGradientTensor->getDimensionSize(concatDimension))
    {
        services::SharedPtr<services::Error> error = services::SharedPtr<services::Error>(new services::Error(
                                                                                              services::ErrorIncorrectSizeOfDimensionInTensor));
        error->addStringDetail(services::ArgumentName, "inputGradient");
        this->_errors->add(error);
        return;
    }
}

/**
 * Returns the layout of the result object for the layer algorithm
 * \return Layout of the result object for the layer algorithm
 */
LayerResultLayout Result::getLayout() const  { return collectionResult; }

size_t Result::getElem(data_management::NumericTablePtr nt, size_t index) const
{
    data_management::BlockDescriptor<int> block;
    nt->getBlockOfRows(0, 1, data_management::readOnly, block);
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
