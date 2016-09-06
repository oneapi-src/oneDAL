/* file: layer_forward.cpp */
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
//  Implementation of neural_networks forward layer methods.
//--
*/

#include "layer_forward_types.h"
#include "layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace forward
{
namespace interface1
{
/**
 * Constructs input objects for the forward layer of neural network
 * \param[in] nElements     Number of input objects for the forward layer
 */
Input::Input(size_t nElements) : InputIface(nElements)
{
    Argument::set(inputLayerData, services::SharedPtr<LayerData>(new LayerData()));
};

/**
 * Returns input Tensor of the layer algorithm
 * \param[in] id    Identifier of the input tensor
 * \return          %Input tensor that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Input::get(forward::InputId id) const
{
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the layer algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const services::SharedPtr<data_management::Tensor> &ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns input InputLayerData of the layer algorithm
* \param[in] id    Identifier of the input object
* \return          %Input InputLayerData that corresponds to the given identifier
*/
services::SharedPtr<LayerData> Input::get(forward::InputLayerDataId id) const
{
    return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
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
 * Checks an input object for the layer algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 4) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    if (!data_management::checkTensor(get(data).get(), this->_errors.get(), dataStr())) { return; }
}

/**
 * Returns the layout of the input object for the layer algorithm
 * \return Layout of the input object for the layer algorithm
 */
LayerInputLayout Input::getLayout()
{
    return tensorInput;
}

/**
 * Returns dimensions of weights tensor
 * \return Dimensions of weights tensor
 */
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}

/**
 * Returns dimensions of biases tensor
 * \return Dimensions of biases tensor
 */
const services::Collection<size_t> Input::getBiasesSizes(const layers::Parameter *parameter) const
{
    return services::Collection<size_t>();
}

/**
 * Adds tensor with data to the input object of the layer algorithm
 * \param[in] dataTensor    Tensor with data
 * \param[in] index         Index of the tensor with data
 */
void Input::addData(const services::SharedPtr<data_management::Tensor> &dataTensor, size_t index)
{
    set(layers::forward::data, dataTensor);
}

/** \brief Constructor */
Result::Result() : daal::algorithms::Result(2) {};

/**
* Returns collection of dimensions of layer output
* \param[in] inputSize Collection of input tensors dimensions
* \param[in] par       Parameters of the algorithm
* \param[in] method    Method of the algorithm
* \return    Collection of dimensions of layer output
*/
services::Collection<size_t> Result::getValueSize(const services::Collection< services::Collection<size_t> > &inputSize,
                                                  const daal::algorithms::Parameter *par, const int method)
{
    return services::Collection<size_t>();
}

/**
* Returns collection of dimensions of layer output
* \param[in] inputSize   Collection of input tensor dimensions
* \param[in] par         Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of layer output
*/
services::Collection< services::Collection<size_t> > Result::getValueCollectionSize(const services::Collection<size_t> &inputSize,
                                                                                    const daal::algorithms::Parameter *par, const int method)
{
    return services::Collection< services::Collection<size_t> >();
}

/**
 * Returns result of the layer algorithm
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns result of the layer algorithm
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
services::SharedPtr<LayerData> Result::get(ResultLayerDataId id) const
{
    return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the layer algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const services::SharedPtr<data_management::Tensor> &ptr)
{
    Argument::set(id, ptr);
}

/**
* Sets the result of the layer algorithm
* \param[in] id    Identifier of the result
* \param[in] ptr   Pointer to the result
*/
void Result::set(ResultLayerDataId id, const services::SharedPtr<LayerData> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result object for the layer algorithm
 * \param[in] input         %Input of the algorithm
 * \param[in] parameter     %Parameter of algorithm
 * \param[in] method        Computation method of the algorithm
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    if (!data_management::checkTensor(get(value).get(), this->_errors.get(), valueStr())) { return; }
    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        services::SharedPtr<LayerData> layerData = get(resultForBackward);
        if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }
    }
}

/**
 * Returns the layout of the result object for the layer algorithm
 * \return Layout of the result object for the layer algorithm
 */
LayerResultLayout Result::getLayout()
{
    return tensorResult;
}

/**
 * Returns resulting value of the layer algorithm
 * \param[in] index Index of the tensor with value
 * \return Resulting value that corresponds to the given index
 */
services::SharedPtr<data_management::Tensor> Result::getValue(size_t index) const
{
    return get(layers::forward::value);
}

}// namespace interface1
}// namespace forward
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
