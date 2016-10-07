/* file: split_layer_backward.cpp */
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
//  Implementation of split calculation algorithm and types methods.
//--
*/

#include "split_layer_backward_types.h"
#include "split_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace backward
{
namespace interface1
{
/** \brief Default constructor */
Input::Input()
{
    set(inputGradientCollection, services::SharedPtr<LayerData>(new LayerData()));
}

/**
 * Returns a tensor with a given index from the collection of input tensors
 * \param[in] id    Identifier of the collection of input tensors
 * \param[in] index Index of the tensor to be returned
 * \return          Pointer to the table with the input tensor
 */
services::SharedPtr<data_management::Tensor> Input::get(InputLayerDataId id, size_t index) const
{
    services::SharedPtr<layers::LayerData> layerData = get(id);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[index]);
}

/**
 * Returns input Tensor of the layer algorithm
 * \param[in] id    Identifier of the input tensor
 * \return          %Input tensor that corresponds to the given identifier
 */
services::SharedPtr<LayerData> Input::get(InputLayerDataId id) const
{
    return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the backward split layer
 * \param[in] id     Identifier of the input object
 * \param[in] value  Pointer to the input object
 * \param[in] index  Index of the tensor to be set
 */
void Input::set(InputLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
{
    services::SharedPtr<layers::LayerData> layerData = get(id);
    (*layerData)[index] = value;
}

/**
* Sets input for the layer algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(InputLayerDataId id, const services::SharedPtr<LayerData> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Adds tensor with input gradient to the input object of the backward split layer
 * \param[in] igTensor  Tensor with input gradient
 * \param[in] index     Index of the tensor with input gradient
 */
void Input::addInputGradient(const services::SharedPtr<data_management::Tensor> &igTensor, size_t index)
{
    services::SharedPtr<LayerData> layerData = get(inputGradientCollection);

    size_t nInputs = layerData->size();
    (*(layerData))[nInputs] = igTensor;
    set(inputGradientCollection, layerData);
}

/**
 * Sets input structure retrieved from the result of the forward layer
 * \param[in] result Result of the forward layer
 */
void Input::setInputFromForward(services::SharedPtr<layers::forward::Result> result)
{}

/**
 * Checks an input object of the backward split layer
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return; }

    if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }
    services::SharedPtr<LayerData> layerData = get(inputGradientCollection);
    size_t nInputs = parameter->nInputs;

    if (layerData->size() != nInputs) {this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; };
    services::SharedPtr<data_management::Tensor> inputTensor0 = get(inputGradientCollection, 0);

    if (!data_management::checkTensor(inputTensor0.get(), this->_errors.get(), inputGradientCollectionStr()))
    {
        size_t inx = this->_errors->size();
        (*this->_errors->getErrors() )[inx - 1]->addIntDetail(services::ArgumentName, (int)0);
        return;
    }

    services::Collection<size_t> dims0 = inputTensor0->getDimensions();

    for (size_t i = 1; i < nInputs; i++)
    {
        if (!data_management::checkTensor(get(inputGradientCollection, i).get(), this->_errors.get(), inputGradientCollectionStr(), &dims0))
        {
            size_t inx = this->_errors->size();
            (*this->_errors->getErrors() )[inx - 1]->addIntDetail(services::ArgumentName, (int)i);
            return;
        }
    }
}

/**
* Returns the layout of the input object for the layer algorithm
* \return Layout of the input object for the layer algorithm
*/
LayerInputLayout Input::getLayout() const { return collectionInput; }

    /** \brief Default constructor */
Result::Result() : layers::backward::Result() {};

/**
 * Checks the result of the backward split layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return; }

    const Input *algInput = static_cast<const Input *>(input);

    if(Argument::size() != 4) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    services::SharedPtr<data_management::Tensor> inputTensor = algInput->get(inputGradientCollection, 0);
    services::Collection<size_t> dims = inputTensor->getDimensions();

    if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(), &dims)) { return; }
}

}// namespace interface1
}// namespace backward
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
