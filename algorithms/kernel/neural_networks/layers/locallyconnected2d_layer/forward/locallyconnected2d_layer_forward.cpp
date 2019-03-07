/* file: locallyconnected2d_layer_forward.cpp */
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
//  Implementation of locally connected calculation algorithm and types methods.
//--
*/

#include "locallyconnected2d_layer_forward_types.h"
#include "locallyconnected2d_layer_types.h"
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
namespace locallyconnected2d
{
namespace forward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOCALLYCONNECTED2D_FORWARD_RESULT_ID);
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
    const Collection<size_t> &inDims = get(layers::forward::data)->getDimensions();

    size_t l3 = (inDims[param->indices.dims[0]] + 2 * param->paddings.size[0] - param->kernelSizes.size[0]) / param->strides.size[0] + 1;
    size_t l4 = (inDims[param->indices.dims[1]] + 2 * param->paddings.size[1] - param->kernelSizes.size[1]) / param->strides.size[1] + 1;

    services::Collection<size_t> wDims;
    wDims << param->nKernels << l3 << l4 << inDims[param->groupDimension] << param->kernelSizes.size[0] << param->kernelSizes.size[1];

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
    const Collection<size_t> &inDims = get(layers::forward::data)->getDimensions();

    size_t l3 = (inDims[param->indices.dims[0]] + 2 * param->paddings.size[0] - param->kernelSizes.size[0]) / param->strides.size[0] + 1;
    size_t l4 = (inDims[param->indices.dims[1]] + 2 * param->paddings.size[1] - param->kernelSizes.size[1]) / param->strides.size[1] + 1;

    Collection<size_t> bDims;
    bDims << param->nKernels << l3 << l4;

    return bDims;
}

/**
 * Checks input object of the forward 2D locally connected layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, layers::forward::Input::check(parameter, method));

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr dataTensor = get(layers::forward::data);
    data_management::TensorPtr wTensor = get(layers::forward::weights);
    data_management::TensorPtr bTensor = get(layers::forward::biases);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(dataTensor.get(), dataStr()));

    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();

    int a = (int)dataDims[algParameter->indices.dims[0]] + 2 * algParameter->paddings.size[0] - (int)algParameter->kernelSizes.size[0];
    int b = (int)dataDims[algParameter->indices.dims[1]] + 2 * algParameter->paddings.size[1] - (int)algParameter->kernelSizes.size[1];

    DAAL_CHECK(a > 0 || b > 0, ErrorIncorrectParameter);
    DAAL_CHECK_EX(dataDims[algParameter->groupDimension] % algParameter->nGroups == 0 ||
                  algParameter->nKernels % algParameter->nGroups == 0, ErrorIncorrectParameter, ParameterName, nGroupsStr());

    size_t nDims = dataDims.size();

    if( nDims != 4 ) return services::Status( services::ErrorIncorrectNumberOfDimensionsInTensor );

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
 * Returns dimensions of value tensor
 * \return Dimensions of value tensor
 */
const services::Collection<size_t> Result::getValueSize(const services::Collection<size_t> &inputSize,
                                                        const daal::algorithms::Parameter *parameter, const int method) const
{
    const Parameter *param =  static_cast<const Parameter * >(parameter);

    size_t l3 = (inputSize[param->indices.dims[0]] + 2 * param->paddings.size[0] - param->kernelSizes.size[0]) / param->strides.size[0] + 1;
    size_t l4 = (inputSize[param->indices.dims[1]] + 2 * param->paddings.size[1] - param->kernelSizes.size[1]) / param->strides.size[1] + 1;

    services::Collection<size_t> valueDims;
    for(size_t i = 0; i < inputSize.size(); i++)
    {
        if(i == param->indices.dims[0])      { valueDims << l3; }
        else if(i == param->indices.dims[1]) { valueDims << l4; }
        else if(i == param->groupDimension)  { valueDims << param->nKernels; }
        else { valueDims.push_back( inputSize[i] ); }
    }

    return valueDims;
}

/**
 * Sets the result that is used in backward 2D locally connected layer
 * \param[in] input     Pointer to an object containing the input data
 */
services::Status Result::setResultForBackward(const daal::algorithms::Input *input)
{
    const Input *in = static_cast<const Input * >(input);
    set(auxData, in->get(layers::forward::data));
    set(auxWeights, in->get(layers::forward::weights));
    return services::Status();
}

/**
 * Returns the result of forward 2D locally connected layer
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
data_management::TensorPtr Result::get(LayerDataId id) const
{
    LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(!layerData)
        return data_management::TensorPtr();
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of forward 2D locally connected layer
 * \param[in] id     Identifier of the result
 * \param[in] value  Result
 */
void Result::set(LayerDataId id, const data_management::TensorPtr &value)
{
    LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::forward::resultForBackward));
    if(layerData)
        (*layerData)[id] = value;
}

/**
 * Checks the result of the forward 2D locally connected layer
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
    if (!layerData && algParameter->predictionStage == false) return services::Status(services::ErrorNullLayerData);

    data_management::TensorPtr dataTensor  = algInput->get(layers::forward::data);
    data_management::TensorPtr valueTensor = get(layers::forward::value);

    const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
    const services::Collection<size_t>     wDims = algInput->getWeightsSizes(algParameter);
    const services::Collection<size_t>   valDims = getValueSize(dataDims, algParameter, defaultDense);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(valueTensor.get(), valueStr(), &valDims));
    if (algParameter->predictionStage == false)
    {
        DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxData).get(), auxDataStr(), &dataDims));
        if(algInput->get(layers::forward::weights))
        {
            DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxWeights).get(), auxWeightsStr(), &wDims));
        }
    }
    return s;
}

}// namespace interface1
}// namespace forward
}// namespace locallyconnected2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
