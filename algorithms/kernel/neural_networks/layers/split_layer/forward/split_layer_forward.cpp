/* file: split_layer_forward.cpp */
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

#include "split_layer_forward_types.h"
#include "split_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

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
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPLIT_FORWARD_RESULT_ID);
/** \brief Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/** \brief Default constructor */
Result::Result() : layers::forward::Result() {};

/**
 * Returns a tensor with a given index from the result
 * \param[in] id    Identifier of the collection of input tensors
 * \param[in] index Index of the tensor to be returned
 * \return          Pointer to the table with the input tensor
 */
data_management::TensorPtr Result::get(ResultLayerDataId id, size_t index) const
{
    LayerDataPtr resCollection = get(id);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*resCollection)[index]);
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
 * Sets the result of the forward split layer
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 *
 * \return Status of computations
 */
void Result::set(ResultLayerDataId id, const LayerDataPtr &value)
{
    Argument::set(id, value);
}

/**
 * Sets the result of the forward split layer
 * \param[in] id      Identifier of the result
 * \param[in] value   Result
 * \param[in] index   Index of the result
 *
 * \return Status of computations
 */
void Result::set(ResultLayerDataId id, const data_management::TensorPtr &value, size_t index)
{
    LayerDataPtr layerData = this->get(id);
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
data_management::TensorPtr Result::getValue(size_t index) const
{
    return get(valueCollection, index);
}

/**
 * Checks the result of the forward split layer
 * \param[in] input   %Input object for the algorithm
 * \param[in] par     %Parameter of the algorithm
 * \param[in] method  Computation method
 *
 * \return Status of computations
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == 2, services::ErrorIncorrectNumberOfInputNumericTables);

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *parameter = static_cast<const Parameter *>(par);
    const size_t nOutputs = parameter->nOutputs;

    data_management::TensorPtr inputTensor = algInput->get(layers::forward::data);
    services::Status s;
    DAAL_CHECK_STATUS(s, data_management::checkTensor(inputTensor.get(), dataStr()));

    const services::Collection<size_t> &inputDims = inputTensor->getDimensions();

    for (size_t i = 0; i < nOutputs; i++)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(valueCollection, i).get(), valueCollectionStr(), &inputDims));
    }
    return s;
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
    const size_t nOutputs = par->nOutputs;

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
