/* file: split_layer_forward.cpp */
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

#include "split_layer_forward_types.h"
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
namespace forward
{
namespace interface1
{
/** \brief Default constructor */
Input::Input() {};

/** \brief Default constructor */
Result::Result() : layers::forward::Result() {};

/**
 * Returns a tensor with a given index from the result
 * \param[in] id    Identifier of the collection of input tensors
 * \param[in] index Index of the tensor to be returned
 * \return          Pointer to the table with the input tensor
 */
services::SharedPtr<data_management::Tensor> Result::get(ResultLayerDataId id, size_t index) const
{
    services::SharedPtr<LayerData> resCollection = get(id);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*resCollection)[index]);
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
 * Sets the result of the forward split layer
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 */
void Result::set(ResultLayerDataId id, const services::SharedPtr<LayerData> &value)
{
    Argument::set(id, value);
}

/**
 * Sets the result of the forward split layer
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 * \param[in] index   Index of the result
 */
void Result::set(ResultLayerDataId id, const services::SharedPtr<data_management::Tensor> &value, size_t index)
{
    services::SharedPtr<LayerData> layerData = this->get(id);
    (*layerData)[index] = value;
}

/**
 * Returns the layout of the result object for the layer algorithm
 * \return Layout of the result object for the layer algorithm
 */
LayerResultLayout Result::getLayout()  { return collectionResult; }

/**
 * Returns resulting value of the forward split layer
 * \param[in] index Index of the tensor with value
 * \return Resulting value that corresponds to the given index
 */
services::SharedPtr<data_management::Tensor> Result::getValue(size_t index) const
{
    return get(valueCollection, index);
}

/**
 * Checks the result of the forward split layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *parameter = static_cast<const Parameter *>(par);
    size_t nOutputs = parameter->nOutputs;

    services::SharedPtr<data_management::Tensor> inputTensor = algInput->get(layers::forward::data);
    if (!data_management::checkTensor(inputTensor.get(), this->_errors.get(), dataStr())) { return; }

    services::Collection<size_t> inputDims = inputTensor->getDimensions();

    for (size_t i = 0; i < nOutputs; i++)
    {
        if (!data_management::checkTensor(get(valueCollection, i).get(), this->_errors.get(), valueCollectionStr(), &inputDims)) { return; }
    }
}

/**
* Returns collection of dimensions of split layer output
* \param[in] inputSize   Collection of input tensor dimensions
* \param[in] par         Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of split layer output
*/
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    return services::Collection<size_t>();
}

/**
* Returns collection of dimensions of split layer output
* \param[in] inputSize   Collection of input tensor dimensions
* \param[in] parameter   Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of split layer output
*/
services::Collection< services::Collection<size_t> > Result::getValueCollectionSize(const services::Collection<size_t> &inputSize,
                                                                            const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *par = static_cast<const Parameter *>(parameter);
    size_t nOutputs = par->nOutputs;

    services::Collection<services::Collection<size_t> > dimsCollection;

    for (size_t i = 0; i < nOutputs; i++)
    {
        dimsCollection.push_back(inputSize);
    }

    return dimsCollection;
}

}// namespace interface1
}// namespace forward
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
