/* file: concat_layer_forward.cpp */
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
/** \brief Default constructor */
Input::Input() {};

/**
* Returns input Tensor of the forward concat layer
* \param[in] id       Identifier of the input object
* \param[in] index    Index of the input object
* \return             %Input tensor that corresponds to the given identifier
*/
services::SharedPtr<data_management::Tensor> Input::get(layers::forward::InputLayerDataId id, size_t index) const
{
    services::SharedPtr<LayerData> layerData = get(id);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[index]);
}

/**
* Returns input LayerData of the forward concat layer
* \param[in] id    Identifier of the input object
* \return          %Input InputLayerData that corresponds to the given identifier
*/
services::SharedPtr<LayerData> Input::get(layers::forward::InputLayerDataId id) const
{
    return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
}

/**
* Sets input for the forward concat layer
* \param[in] id      Identifier of the input object
* \param[in] value   Pointer to the object
*/
void Input::set(layers::forward::InputLayerDataId id, const services::SharedPtr<LayerData> &value)
{
    Argument::set(id, value);
}

/**
* Sets input for the forward concat layer
* \param[in] id      Identifier of the input object
* \param[in] value   Pointer to the object
* \param[in] index   Index of the input object
*/
void Input::set(layers::forward::InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
{
    services::SharedPtr<LayerData> layerData = get(id);
    (*layerData)[index] = value;
}

/**
 * Adds tensor with data to the input object of the forward concat layer
 * \param[in] dataTensor    Tensor with data
 * \param[in] index         Index of the tensor with data
 */
void Input::addData(const services::SharedPtr<data_management::Tensor> &dataTensor, size_t index)
{
    services::SharedPtr<LayerData> layerData = get(layers::forward::inputLayerData);
    if (!layerData)
    {
        layerData = services::SharedPtr<LayerData>(new LayerData());
    }
    size_t nInputs = layerData->size();
    (*(layerData))[nInputs] = dataTensor;
    set(layers::forward::inputLayerData, layerData);
}

/**
 * Erases input data tensor from the input of the forward layer
 */
void Input::eraseInputData()
{
    set(layers::forward::inputLayerData, services::SharedPtr<LayerData>());
}

/**
* Checks an input object for the forward concat layer
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method of the algorithm
*/
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    size_t concatDimension = parameter->concatDimension;
    services::SharedPtr<LayerData> layerData = get(layers::forward::inputLayerData);
    if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

    size_t nInputs = layerData->size();
    if (nInputs == 0) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }
    services::SharedPtr<data_management::Tensor> layerDataTensor0 = get(layers::forward::inputLayerData, 0);

    if (!data_management::checkTensor(layerDataTensor0.get(), this->_errors.get(), inputLayerDataStr())) { return; }

    services::Collection<size_t> dims = layerDataTensor0->getDimensions();

    if (concatDimension > dims.size() - 1) {this->_errors->add(services::ErrorIncorrectParameter); return; }

    for (size_t i = 1; i < nInputs; i++)
    {
        services::SharedPtr<data_management::Tensor> layerDataTensor = get(layers::forward::inputLayerData, i);
        dims[concatDimension] = layerDataTensor->getDimensionSize(concatDimension);

        if (!data_management::checkTensor(layerDataTensor.get(), this->_errors.get(), inputLayerDataStr(), &dims)) { return; }
    }
}

/**
* Returns the layout of the input object for the layer algorithm
* \return Layout of the input object for the layer algorithm
*/
LayerInputLayout Input::getLayout()  { return collectionInput; }

    /** \brief Default constructor */
Result::Result() : layers::forward::Result() {};

/**
* Sets the result of the forward concat layer
* \param[in] id      Identifier of the result
* \param[in] value   Pointer to the result
*/
void Result::set(LayerDataId id, const data_management::NumericTablePtr &value)
{
    (*get(layers::forward::resultForBackward))[id] = value;
}

/**
* Returns input object of the forward concat layer
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
data_management::NumericTablePtr Result::get(layers::concat::LayerDataId id) const
{
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>
           ((*get(layers::forward::resultForBackward))[id]);
}

/**
* Returns collection of dimensions of concat layer output
* \param[in] inputSize   Collection of input tensors dimensions
* \param[in] par         Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of concat layer output
*/
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    return services::Collection<size_t>();
}

/**
* Returns collection of dimensions of concat layer output
* \param[in] inputSize   Collection of input tensors dimensions
* \param[in] parameter   Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of concat layer output
*/
services::Collection<size_t> Result::getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                          const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    size_t nInputs = inputSize.size();
    size_t concatDimension = par->concatDimension;

    size_t sum = 0;
    for (size_t i = 0; i < nInputs; i++)
    {
        sum += inputSize[i][concatDimension];
    }

    services::Collection<size_t> dimsCollection = inputSize[0];
    dimsCollection[concatDimension] = sum;

    return dimsCollection;
}

/**
* Checks the result object for the layer algorithm
* \param[in] input         %Input of the algorithm
* \param[in] parameter     %Parameter of algorithm
* \param[in] method        Computation method of the algorithm
*/
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                   int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t concatDimension = algParameter->concatDimension;

    if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
    if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }

    services::SharedPtr<LayerData> inputLayerData = algInput->get(layers::forward::inputLayerData);
    size_t inputSize = inputLayerData->size();
    data_management::NumericTablePtr dimsNT = get(auxInputDimensions);
    if (dimsNT->getNumberOfColumns() != inputSize) { this->_errors->add(services::ErrorIncorrectNumberOfColumnsInOutputNumericTable); return; }
    if (dimsNT->getNumberOfRows() != 1) { this->_errors->add(services::ErrorIncorrectNumberOfRowsInOutputNumericTable); return; }

    size_t sum = 0;
    for (size_t i = 0; i < inputSize; i++)
    {
        services::SharedPtr<data_management::Tensor> inputTensor = algInput->get(layers::forward::inputLayerData, i);
        size_t dim = inputTensor->getDimensionSize(concatDimension);

        sum += dim;
    }

    services::SharedPtr<data_management::Tensor> valueTensor = get(layers::forward::value);
    services::Collection<size_t> dims = algInput->get(layers::forward::inputLayerData, 0)->getDimensions();
    dims[concatDimension] = sum;

    if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr(), &dims)) { return; }
}

}// namespace interface1
}// namespace forward
}// namespace concat
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
