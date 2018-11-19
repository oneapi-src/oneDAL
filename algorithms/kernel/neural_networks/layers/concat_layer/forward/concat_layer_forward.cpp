/* file: concat_layer_forward.cpp */
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
//  Implementation of concat calculation algorithm and types methods.
//--
*/

#include "concat_layer_forward_types.h"
#include "concat_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONCAT_FORWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
* Returns input Tensor of the forward concat layer
* \param[in] id       Identifier of the input object
* \param[in] index    Index of the input object
* \return             %Input tensor that corresponds to the given identifier
*/
TensorPtr Input::get(layers::forward::InputLayerDataId id, size_t index) const
{
    LayerDataPtr layerData = get(id);
    return staticPointerCast<Tensor, SerializationIface>((*layerData)[index]);
}

/**
* Returns input LayerData of the forward concat layer
* \param[in] id    Identifier of the input object
* \return          %Input InputLayerData that corresponds to the given identifier
*/
LayerDataPtr Input::get(layers::forward::InputLayerDataId id) const
{
    return staticPointerCast<LayerData, SerializationIface>(Argument::get(id));
}

/**
* Sets input for the forward concat layer
* \param[in] id      Identifier of the input object
* \param[in] value   Pointer to the object
*/
void Input::set(layers::forward::InputLayerDataId id, const LayerDataPtr &value)
{
    Argument::set(id, value);
}

/**
* Sets input for the forward concat layer
* \param[in] id      Identifier of the input object
* \param[in] value   Pointer to the object
* \param[in] index   Index of the input object
*/
void Input::set(layers::forward::InputLayerDataId id, const TensorPtr &value, size_t index)
{
    LayerDataPtr layerData = get(id);
    (*layerData)[index] = value;
}

/**
 * Adds tensor with data to the input object of the forward concat layer
 * \param[in] dataTensor    Tensor with data
 * \param[in] index         Index of the tensor with data
 */
Status Input::addData(const TensorPtr &dataTensor, size_t index)
{
    LayerDataPtr layerData = get(layers::forward::inputLayerData);
    if (!layerData) layerData = LayerDataPtr(new LayerData());

    const size_t nInputs = layerData->size();
    (*(layerData))[nInputs] = dataTensor;
    set(layers::forward::inputLayerData, layerData);
    return Status();
}

/**
 * Erases input data tensor from the input of the forward layer
 */
Status Input::eraseInputData()
{
    set(layers::forward::inputLayerData, LayerDataPtr());
    return Status();
}

/**
* Checks an input object for the forward concat layer
* \param[in] par     %Parameter of algorithm
* \param[in] method  Computation method of the algorithm
*/
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    const size_t concatDimension = parameter->concatDimension;
    LayerDataPtr layerData = get(layers::forward::inputLayerData);
    DAAL_CHECK(layerData, ErrorNullLayerData);

    const size_t nInputs = layerData->size();
    DAAL_CHECK((nInputs != 0), ErrorIncorrectSizeOfLayerData);
    TensorPtr layerDataTensor0 = get(layers::forward::inputLayerData, 0);

    Status s;
    DAAL_CHECK_STATUS(s, checkTensor(layerDataTensor0.get(), inputLayerDataStr()));

    Collection<size_t> dims = layerDataTensor0->getDimensions();

    DAAL_CHECK((concatDimension <= dims.size() - 1), ErrorIncorrectParameter);

    for (size_t i = 1; i < nInputs; i++)
    {
        TensorPtr layerDataTensor = get(layers::forward::inputLayerData, i);
        dims[concatDimension] = layerDataTensor->getDimensionSize(concatDimension);

        DAAL_CHECK_STATUS(s, checkTensor(layerDataTensor.get(), inputLayerDataStr(), &dims));
    }
    return s;
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
void Result::set(LayerDataId id, const NumericTablePtr &value)
{
    layers::LayerDataPtr layerData =
        staticPointerCast<layers::LayerData, SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(!layerData)
    {
        return;
    }
    //instead of "if":
    // DAAL_CHECK(layerData, ErrorNullLayerData);
    (*layerData)[id] = value;
    // return Status();
}

/**
* Returns input object of the forward concat layer
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
NumericTablePtr Result::get(layers::concat::LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        staticPointerCast<layers::LayerData, SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(!layerData)
    {
        return NumericTablePtr();
    }
    return staticPointerCast<NumericTable, SerializationIface>
           ((*layerData)[id]);
}

/**
* Returns collection of dimensions of concat layer output
* \param[in] inputSize   Collection of input tensors dimensions
* \param[in] par         Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of concat layer output
*/
const Collection<size_t> Result::getValueSize(const Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    return Collection<size_t>();
}

/**
* Returns collection of dimensions of concat layer output
* \param[in] inputSize   Collection of input tensors dimensions
* \param[in] parameter   Parameters of the algorithm
* \param[in] method      Method of the algorithm
* \return    Collection of dimensions of concat layer output
*/
Collection<size_t> Result::getValueSize(const Collection< Collection<size_t> > &inputSize,
                                          const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *par = static_cast<const Parameter *>(parameter);

    const size_t nInputs = inputSize.size();
    const size_t concatDimension = par->concatDimension;

    size_t sum = 0;
    for (size_t i = 0; i < nInputs; i++)
    {
        sum += inputSize[i][concatDimension];
    }

    Collection<size_t> dimsCollection = inputSize[0];
    dimsCollection[concatDimension] = sum;

    return dimsCollection;
}

/**
* Checks the result object for the layer algorithm
* \param[in] input         %Input of the algorithm
* \param[in] parameter     %Parameter of algorithm
* \param[in] method        Computation method of the algorithm
*/
Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
                   int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    const size_t concatDimension = algParameter->concatDimension;

    DAAL_CHECK((Argument::size() == 2), ErrorIncorrectNumberOfInputNumericTables);

    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    DAAL_CHECK((layerData || algParameter->predictionStage), ErrorNullLayerData);

    LayerDataPtr inputLayerData = algInput->get(layers::forward::inputLayerData);
    const size_t inputSize = inputLayerData->size();

    if(!algParameter->predictionStage)
    {
        NumericTablePtr dimsNT = get(auxInputDimensions);
        DAAL_CHECK((dimsNT.get() != 0), ErrorNullNumericTable);
        DAAL_CHECK((dimsNT->getNumberOfColumns() == inputSize), ErrorIncorrectNumberOfColumnsInOutputNumericTable);
        DAAL_CHECK((dimsNT->getNumberOfRows() == 1), ErrorIncorrectNumberOfRowsInOutputNumericTable);
    }
    size_t sum = 0;
    for (size_t i = 0; i < inputSize; i++)
    {
        TensorPtr inputTensor = algInput->get(layers::forward::inputLayerData, i);
        const size_t dim = inputTensor->getDimensionSize(concatDimension);

        sum += dim;
    }

    TensorPtr valueTensor = get(layers::forward::value);
    Collection<size_t> dims = algInput->get(layers::forward::inputLayerData, 0)->getDimensions();
    dims[concatDimension] = sum;

    return checkTensor(get(layers::forward::value).get(), valueStr(), &dims);
}

}// namespace interface1
}// namespace forward
}// namespace concat
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
