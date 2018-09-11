/* file: split_layer_backward.cpp */
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
//  Implementation of split calculation algorithm and types methods.
//--
*/

#include "split_layer_backward_types.h"
#include "split_layer_types.h"
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
namespace split
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPLIT_BACKWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input()
{
    set(inputGradientCollection, LayerDataPtr(new LayerData()));
}

/**
 * Returns a tensor with a given index from the collection of input tensors
 * \param[in] id    Identifier of the collection of input tensors
 * \param[in] index Index of the tensor to be returned
 * \return          Pointer to the table with the input tensor
 */
TensorPtr Input::get(InputLayerDataId id, size_t index) const
{
    layers::LayerDataPtr layerData = get(id);
    return staticPointerCast<Tensor, SerializationIface>((*layerData)[index]);
}

/**
 * Returns input Tensor of the layer algorithm
 * \param[in] id    Identifier of the input tensor
 * \return          %Input tensor that corresponds to the given identifier
 */
LayerDataPtr Input::get(InputLayerDataId id) const
{
    return staticPointerCast<LayerData, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the backward split layer
 * \param[in] id     Identifier of the input object
 * \param[in] value  Pointer to the input object
 * \param[in] index  Index of the tensor to be set
 */
void Input::set(InputLayerDataId id, const TensorPtr &value, size_t index)
{
    layers::LayerDataPtr layerData = get(id);
    (*layerData)[index] = value;
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
 * Adds tensor with input gradient to the input object of the backward split layer
 * \param[in] igTensor  Tensor with input gradient
 * \param[in] index     Index of the tensor with input gradient
 *
 * \return Status of computations
 */
Status Input::addInputGradient(const TensorPtr &igTensor, size_t index)
{
    LayerDataPtr layerData = get(inputGradientCollection);

    const size_t nInputs = layerData->size();
    (*(layerData))[nInputs] = igTensor;
    set(inputGradientCollection, layerData);

    return Status();
}

/**
 * Sets input structure retrieved from the result of the forward layer
 * \param[in] result Result of the forward layer
 */
Status Input::setInputFromForward(layers::forward::ResultPtr result)
{
    return Status();
}

/**
 * Checks an input object of the backward split layer
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return Status(); }

    DAAL_CHECK(Argument::size() == 2, ErrorIncorrectNumberOfInputNumericTables);

    LayerDataPtr layerData = get(inputGradientCollection);
    const size_t nInputs = parameter->nInputs;

    DAAL_CHECK(layerData->size() == nInputs, ErrorIncorrectSizeOfLayerData);
    TensorPtr inputTensor0 = get(inputGradientCollection, 0);

    Status s = checkTensor(inputTensor0.get(), inputGradientCollectionStr());
    if(!s)
    {
        auto e = Error::create(ErrorSplitLayerBackward, ArgumentName, inputGradientCollectionStr());
        e->addIntDetail(ElementInCollection, 0);
        Status s1(e);
        return s1.add(s);
    }

    const Collection<size_t> &dims0 = inputTensor0->getDimensions();

    for (size_t i = 1; i < nInputs; i++)
    {
        s = checkTensor(get(inputGradientCollection, i).get(), inputGradientCollectionStr(), &dims0);
        if (!s)
        {
            auto e = Error::create(ErrorSplitLayerBackward, ArgumentName, inputGradientCollectionStr());
            e->addIntDetail(ElementInCollection, 0);
            Status s1(e);
            return s1.add(s);
        }
    }
    return s;
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
 *
 * \return Status of computations
 */
Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return Status(); }

    const Input *algInput = static_cast<const Input *>(input);

    DAAL_CHECK(Argument::size() == 4, ErrorIncorrectNumberOfInputNumericTables);

    TensorPtr inputTensor = algInput->get(inputGradientCollection, 0);
    const Collection<size_t> &dims = inputTensor->getDimensions();

    return checkTensor(get(layers::backward::gradient).get(), gradientStr(), &dims);
}

}// namespace interface1
}// namespace backward
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
