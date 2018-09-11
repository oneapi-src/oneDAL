/* file: logistic_cross_layer_backward.cpp */
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
//  Implementation of logistic cross calculation algorithm and types methods.
//--
*/

#include "logistic_cross_layer_types.h"
#include "logistic_cross_layer_backward_types.h"
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
namespace loss
{
namespace logistic_cross
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOSS_LOGISTIC_CROSS_BACKWARD_RESULT_ID);
/** Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for the backward logistic cross-entropy layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
TensorPtr Input::get(LayerDataId id) const
{
    LayerDataPtr layerData =
        layers::LayerData::cast<SerializationIface>(Argument::get(layers::backward::inputFromForward));
    if(!layerData)
        return data_management::TensorPtr();
    return Tensor::cast<SerializationIface>((*layerData)[id]);
}

/**
 * Sets an input object for the backward logistic cross-entropy layer
 * \param[in] id      Identifier of the input object
 * \param[in] value   Pointer to the object
 */
void Input::set(LayerDataId id, const TensorPtr &value)
{
    LayerDataPtr layerData =
        layers::LayerData::cast<SerializationIface>(Argument::get(layers::backward::inputFromForward));
    if(!layerData) return;
    (*layerData)[id] = value;
}

/**
 * Checks an input object for the backward logistic cross-entropy layer
 * \param[in] par     Algorithm parameter
 * \param[in] method  Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const layers::Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, loss::backward::Input::check(par, method));

    TensorPtr auxDataTensor = get(auxData);
    TensorPtr auxGroundTruthTensor = get(auxGroundTruth);

    DAAL_CHECK_STATUS(s, checkTensor(auxDataTensor.get(), dataStr()));
    const Collection<size_t> &inputDims = auxDataTensor->getDimensions();

    DAAL_CHECK_STATUS(s, checkTensor(auxGroundTruthTensor.get(), groundTruthStr()));
    const Collection<size_t> &gtDims = auxGroundTruthTensor->getDimensions();

    DAAL_CHECK_EX(auxDataTensor->getSize() == auxGroundTruthTensor->getSize(), ErrorIncorrectSizeOfDimensionInTensor, ParameterName, groundTruthStr());
    DAAL_CHECK_EX(gtDims.size() == 1 || gtDims.size() == inputDims.size() , ErrorIncorrectNumberOfDimensionsInTensor, ParameterName, dataStr());
    DAAL_CHECK_EX(gtDims[0] == inputDims[0] , ErrorIncorrectSizeOfDimensionInTensor, ParameterName, dataStr());
    return s;
}

    /** Default constructor */
Result::Result() : loss::backward::Result() {};

/**
 * Checks the result of the backward logistic cross-entropy layer
 * \param[in] input   %Input object for the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const layers::Parameter *parameter = static_cast<const Parameter *>(par);
    if (!parameter->propagateGradient) { return services::Status(); }

    const Input *algInput = static_cast<const Input *>(input);
    //get expected gradient dimensions
    const Collection<size_t> &gradDims = algInput->get(auxData)->getDimensions();
    return checkTensor(get(layers::backward::gradient).get(), gradientStr(), &gradDims);
}

}// namespace interface1
}// namespace backward
}// namespace logistic_cross
}// namespace loss
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
