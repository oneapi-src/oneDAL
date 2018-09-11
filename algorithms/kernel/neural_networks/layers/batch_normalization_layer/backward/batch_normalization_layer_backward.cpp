/* file: batch_normalization_layer_backward.cpp */
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
//  Implementation of batch normalization calculation algorithm and types methods.
//--
*/

#include "batch_normalization_layer_backward_types.h"
#include "batch_normalization_layer_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_BATCH_NORMALIZATION_BACKWARD_RESULT_ID);
/** Default constructor */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for the backward batch normalization layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(LayerDataId id) const
{
    layers::LayerDataPtr inputData = get(layers::backward::inputFromForward);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*inputData)[id]);
}

/**
 * Sets an input object for the backward batch normalization layer
 * \param[in] id  Identifier of the input object
 * \param[in] ptr Pointer to the object
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &ptr)
{
    layers::LayerDataPtr inputData = get(layers::backward::inputFromForward);
    (*inputData)[id] = ptr;
}

/**
 * Checks an input object for the backward batch normalization layer
 * \param[in] parameter Layer parameter
 * \param[in] method    Computation method
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(parameter, method));

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr inputGradientTensor = get(layers::backward::inputGradient);
    DAAL_CHECK_STATUS(s, data_management::checkTensor(inputGradientTensor.get(), inputGradientStr()));
    const services::Collection<size_t> &dataDims = inputGradientTensor->getDimensions();

    size_t dimension = algParameter->dimension;
    double epsilon = algParameter->epsilon;
    DAAL_CHECK_EX(dimension <= dataDims.size(), services::ErrorIncorrectParameter, services::ParameterName, dimensionStr());
    DAAL_CHECK_EX(epsilon > 0.0 && epsilon < 1.0, services::ErrorIncorrectParameter, services::ParameterName, epsilonStr());

    size_t dimensionSize = dataDims[dimension];
    services::Collection<size_t> auxDims(1);
    auxDims[0] = dimensionSize;

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxData).get(),               auxDataStr(),              &dataDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxWeights).get(),            auxWeightsStr(),            &auxDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxMean).get(),               auxMeanStr(),               &auxDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxStandardDeviation).get(),  auxStandardDeviationStr(),  &auxDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxPopulationMean).get(),     auxPopulationMeanStr(),     &auxDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxPopulationVariance).get(), auxPopulationVarianceStr(), &auxDims));
    return s;
}

/** Default constructor */
Result::Result() {}

/**
 * Checks the result of the backward batch normalization layer
 * \param[in] input     %Input object for the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t dimension = algParameter->dimension;

    data_management::TensorPtr inputGradientTensor = algInput->get(layers::backward::inputGradient);
    const services::Collection<size_t> &gradientDims = inputGradientTensor->getDimensions();

    services::Status s;
    if (algParameter->propagateGradient)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &gradientDims));
    }

    size_t dimensionSize = gradientDims[dimension];
    services::Collection<size_t> derDims(1);
    derDims[0] = dimensionSize;

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::weightDerivatives).get(), weightDerivativesStr(), &derDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::biasDerivatives).get(), biasDerivativesStr(), &derDims));
    return s;
}

}// namespace interface1
}// namespace backward
}// namespace batch_normalization
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
