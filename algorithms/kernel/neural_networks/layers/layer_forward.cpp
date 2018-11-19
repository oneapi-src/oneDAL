/* file: layer_forward.cpp */
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
//  Implementation of neural_networks forward layer methods.
//--
*/

#include "layer_forward_types.h"
#include "layer_types.h"
#include "daal_strings.h"

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

InputIface::InputIface(const InputIface& other) : daal::algorithms::Input(other) {}
Input::Input(const Input& other) : InputIface(other){}

/**
 * Constructs input objects for the forward layer of neural network
 * \param[in] nElements     Number of input objects for the forward layer
 */
Input::Input(size_t nElements) : InputIface(nElements)
{
    Argument::set(inputLayerData, LayerDataPtr(new LayerData()));
};

/**
 * Returns input Tensor of the layer algorithm
 * \param[in] id    Identifier of the input tensor
 * \return          %Input tensor that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(forward::InputId id) const
{
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the layer algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputId id, const data_management::TensorPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns input InputLayerData of the layer algorithm
* \param[in] id    Identifier of the input object
* \return          %Input InputLayerData that corresponds to the given identifier
*/
LayerDataPtr Input::get(forward::InputLayerDataId id) const
{
    return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets input for the layer algorithm
 * \param[in] id    Identifier of the input object
 * \param[in] ptr   Pointer to the object
 */
void Input::set(InputLayerDataId id, const LayerDataPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks an input object for the layer algorithm
 * \param[in] par     %Parameter of algorithm
 * \param[in] method  Computation method of the algorithm
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    if(Argument::size() != 4) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);

    return data_management::checkTensor(get(data).get(), dataStr());
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
services::Status Input::addData(const data_management::TensorPtr &dataTensor, size_t index)
{
    set(layers::forward::data, dataTensor);
    return services::Status();
}

/** \brief Constructor */
Result::Result() : daal::algorithms::Result(lastResultLayerDataId + 1) {};

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
data_management::TensorPtr Result::get(ResultId id) const
{
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Returns result of the layer algorithm
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
LayerDataPtr Result::get(ResultLayerDataId id) const
{
    return services::staticPointerCast<LayerData, data_management::SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of the layer algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the result
 */
void Result::set(ResultId id, const data_management::TensorPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Sets the result of the layer algorithm
* \param[in] id    Identifier of the result
* \param[in] ptr   Pointer to the result
*/
void Result::set(ResultLayerDataId id, const LayerDataPtr &ptr)
{
    Argument::set(id, ptr);
}

/**
 * Checks the result object for the layer algorithm
 * \param[in] input         %Input of the algorithm
 * \param[in] parameter     %Parameter of algorithm
 * \param[in] method        Computation method of the algorithm
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    if(Argument::size() != 2) return services::Status(services::ErrorIncorrectNumberOfInputNumericTables);

    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(value).get(), valueStr()));
    const layers::Parameter *par = static_cast<const layers::Parameter * >(parameter);
    if(!par->predictionStage)
    {
        LayerDataPtr layerData = get(resultForBackward);
        if (!layerData) return services::Status(services::ErrorNullLayerData);
    }
    return s;
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
data_management::TensorPtr Result::getValue(size_t index) const
{
    return get(layers::forward::value);
}

}// namespace interface1
}// namespace forward
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
