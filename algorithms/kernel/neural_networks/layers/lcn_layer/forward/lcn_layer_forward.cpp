/* file: lcn_layer_forward.cpp */
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
//  Implementation of lcn calculation algorithm and types methods.
//--
*/

#include "lcn_layer_forward_types.h"
#include "lcn_layer_types.h"
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
namespace lcn
{
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_LCN_FORWARD_RESULT_ID);
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
* Checks input object of the forward local contrast normalization layer
* \param[in] parameter %Parameter of layer
* \param[in] method    Computation method of the layer
*/
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Input::check(parameter, method));

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr dataTensor = get(layers::forward::data);
    DAAL_CHECK_STATUS(s, data_management::checkTensor(dataTensor.get(), dataStr()));

    size_t nDims = dataTensor->getNumberOfDimensions();

    if( nDims != 4 ) return( services::ErrorIncorrectNumberOfDimensionsInTensor );
    return services::Status();
}
/**
 * Default constructor
 */
Result::Result() {}

/**
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
        const daal::algorithms::Parameter *par, const int method) const
{
    return inputSize;

}

/**
 * Returns the result of forward local contrast normalization layer
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
 * Sets the result of forward local contrast normalization layer
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
 * Checks the result of the forward local contrast normalization layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Result::check(input, par, method));

    services::SharedPtr<services::Error> error;

    const Input     *algInput     = static_cast<const Input *>(input);
    const Parameter *algParameter = static_cast<const Parameter *>(par);

    LayerDataPtr layerData = get(layers::forward::resultForBackward);
    if (!layerData) return services::Status(services::ErrorNullLayerData);

    const services::Collection<size_t> dataDims  = algInput->get(layers::forward::data)->getDimensions();
    services::Collection<size_t> sigmaDims;
    getSigmaDimensions(algInput, algParameter, sigmaDims);

    services::Collection<size_t> cDims;
    getCDimensions(algInput, algParameter, cDims);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::forward::value).get(), valueStr(), &dataDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxCenteredData).get(), auxCenteredDataStr(), &dataDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxC).get(), auxCStr(), &cDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxInvMax).get(), auxInvMaxStr(), &sigmaDims));
    if(!algParameter->predictionStage)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxSigma).get(), auxSigmaStr(), &sigmaDims));
    }
    return s;
}

void Result::getSigmaDimensions(const Input *in, const Parameter *algParameter, services::Collection<size_t> &sigmaDims) const
{
    sigmaDims = in->get(layers::forward::data)->getDimensions();

    if(algParameter->sumDimension)
    {
        data_management::NumericTablePtr dimensionTable = algParameter->sumDimension;

        data_management::BlockDescriptor<int> block;
        dimensionTable->getBlockOfRows(0, 1, data_management::readOnly, block);
        int *dataInt = block.getBlockPtr();
        size_t dim = dataInt[0];
        dimensionTable->releaseBlockOfRows(block);
        sigmaDims.erase(dim);
    }
}

void Result::getCDimensions(const Input *in, const Parameter *algParameter, services::Collection<size_t> &cDims) const
{
    getSigmaDimensions(in, algParameter, cDims);

    if(algParameter->sumDimension)
    {
        cDims.erase(algParameter->indices.dims[1] - 1);
        cDims.erase(algParameter->indices.dims[0] - 1);
    }
    else
    {
        cDims.erase(algParameter->indices.dims[1]);
        cDims.erase(algParameter->indices.dims[0]);
    }
}


}// namespace interface1
}// namespace forward
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
