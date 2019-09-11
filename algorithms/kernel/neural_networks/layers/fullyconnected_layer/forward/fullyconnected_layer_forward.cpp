/* file: fullyconnected_layer_forward.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of fullyconnected calculation algorithm and types methods.
//--
*/

#include "fullyconnected_layer_forward_types.h"
#include "fullyconnected_layer_types.h"
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
namespace fullyconnected
{
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_FULLYCONNECTED_FORWARD_RESULT_ID);
/**
 * Default constructor
 */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns dimensions of weights tensor
 * \return Dimensions of weights tensor
 */
const services::Collection<size_t> Input::getWeightsSizes(const layers::Parameter *parameter) const
{
    using daal::services::Collection;

    const Parameter *param =  static_cast<const Parameter *>(parameter);

    Collection<size_t> wDims = get(layers::forward::data)->getDimensions();
    wDims[0] = param->nOutputs;

    return wDims;
}

/**
 * Returns dimensions of biases tensor
 * \return Dimensions of biases tensor
 */
const services::Collection<size_t> Input::getBiasesSizes(const layers::Parameter *parameter) const
{
    using daal::services::Collection;

    const Parameter *param =  static_cast<const Parameter *>(parameter);
    size_t m = param->nOutputs;

    Collection<size_t> bDims;
    bDims.push_back( m );

    return bDims;
}

/**
 * Checks input object of the forward fully-connected layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Input::check(parameter, method));

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr wTensor = get(layers::forward::weights);
    data_management::TensorPtr bTensor = get(layers::forward::biases);

    if( wTensor )
    {
        services::Collection<size_t> wDims = getWeightsSizes(algParameter);
        DAAL_CHECK_STATUS(s, data_management::checkTensor(wTensor.get(), weightsStr(), &wDims));
    }

    if( bTensor )
    {
        services::Collection<size_t> bDims = getBiasesSizes(algParameter);
        DAAL_CHECK_STATUS(s, data_management::checkTensor(bTensor.get(), biasesStr(), &bDims));
    }
    return s;
}

/**
 * Default constructor
 */
Result::Result() {}

/**
 * Sets the result that is used in backward fully-connected layer
 * \param[in] input     Pointer to an object containing the input data
 */
services::Status Result::setResultForBackward(const daal::algorithms::Input *input)
{
    const Input *in = static_cast<const Input *>(input);
    set(auxData,    in->get(layers::forward::data));
    set(auxWeights, in->get(layers::forward::weights));
    return services::Status();
}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *par, const int method) const
{
    const Parameter *param =  static_cast<const Parameter *>(par);

    services::Collection<size_t> valueDims;
    valueDims.push_back(inputSize[0]);
    valueDims.push_back(param->nOutputs);
    return valueDims;
}

/**
 * Returns the result of forward fully-connected layer
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
data_management::TensorPtr Result::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(!layerData)
        return data_management::TensorPtr();
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of forward fully-connected layer
 * \param[in] id     Identifier of the result
 * \param[in] value  Result
 */
void Result::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(layerData)
        (*layerData)[id] = value;
}

/**
 * Checks the result of the forward fully-connected layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, par, method));

    const Input     *algInput     = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(par);

    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    if (!layerData && algParameter->predictionStage == false) return services::Status(services::ErrorNullLayerData);

    data_management::TensorPtr dataTensor  = algInput->get(layers::forward::data);

    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
    const services::Collection<size_t>     wDims = algInput->getWeightsSizes(algParameter);
    const services::Collection<size_t>   valDims = getValueSize(dataDims, algParameter, defaultDense);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::forward::value).get(), valueStr(), &valDims));

    if(!algParameter->predictionStage)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxData).get(), auxDataStr(), &dataDims));
        if(algParameter->propagateGradient)
        {
            if(algInput->get(layers::forward::weights))
            {
                DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxWeights).get(), auxWeightsStr(), &wDims));
            }
        }
    }
    return s;
}

}// namespace interface1
}// namespace forward
}// namespace fullyconnected
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
