/* file: layer_backward.cpp */
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
//  Implementation of neural_networks backward layer methods.
//--
*/

#include "layer_backward_types.h"
#include "layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace backward
{
namespace interface1
{
/** \brief Constructor */
Input::Input() : InputIface(2)
{
    set(inputFromForward, services::SharedPtr<LayerData>(new LayerData()));
}

/**
 * Returns input Tensor of the layer algorithm
 * \param[in] id    Identifier of the input tensor
 * \return          %Input tensor that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Input::get(InputId id) const
{
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
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
 * Sets input for the layer algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const services::SharedPtr<data_management::Tensor> &ptr)
{
    Argument::set(id, ptr);
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
 * Adds tensor with input gradient to the input object of the layer algorithm
 * \param[in] igTensor  Tensor with input gradient
 * \param[in] index     Index of the tensor with input gradient
 */
void Input::addInputGradient(const services::SharedPtr<data_management::Tensor> &igTensor, size_t index)
{
    set(layers::backward::inputGradient, igTensor);
}

/**
 * Checks an input object for the layer algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
void Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    if (!data_management::checkTensor(get(inputGradient).get(), this->_errors.get(), inputGradientStr())) { return; }

    services::SharedPtr<LayerData> layerData = get(inputFromForward);
    if (!layerData) { this->_errors->add(services::ErrorNullLayerData); return; }
}

/**
 * Returns the layout of the input object for the layer algorithm
 * \return Layout of the input object for the layer algorithm
 */
LayerInputLayout Input::getLayout() const { return tensorInput; }

    /**
 * Returns result of the layer algorithm
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
}

/** \brief Constructor */
Result::Result() : daal::algorithms::Result(4)
{
    Argument::set(resultLayerData, services::SharedPtr<LayerData>(new LayerData()));
};

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
* Returns result InputLayerData of the layer algorithm
* \param[in] id    Identifier of the result object
* \return          Resulting InputLayerData that corresponds to the given identifier
*/
services::SharedPtr<LayerData> Result::get(backward::ResultLayerDataId id) const
{
    return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets result for the layer algorithm
 * \param[in] id    Identifier of the result object
 * \param[in] ptr   Pointer to the object
 */
void Result::set(ResultLayerDataId id, const services::SharedPtr<LayerData> &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result object for the layer algorithm
 * \param[in] input         %Input of algorithm
 * \param[in] parameter     %Parameter of algorithm
 * \param[in] method        Computation method of the algorithm
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    if(Argument::size() != 4) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    if (!data_management::checkTensor(get(gradient).get(), this->_errors.get(), gradientStr())) { return; }
}

/**
 * Returns the layout of the result object for the layer algorithm
 * \return Layout of the result object for the layer algorithm
 */
LayerResultLayout Result::getLayout() const { return tensorResult; }

/**
 * Returns resulting gradient of the layer algorithm
 * \param[in] index Index of the tensor with gradient
 * \return Resulting gradient that corresponds to the given index
 */
services::SharedPtr<data_management::Tensor> Result::getGradient(size_t index) const
{
    return get(layers::backward::gradient);
}

}// namespace interface1
}// namespace backward
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
