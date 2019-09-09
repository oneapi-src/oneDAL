/* file: lcn_layer_backward.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of lcn calculation algorithm and types methods.
//--
*/

#include "lcn_layer_backward_types.h"
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
namespace backward
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_LAYERS_LCN_BACKWARD_RESULT_ID);
/**
 * Default constructor
 */
Input::Input() {};
Input::Input(const Input& other) : super(other) {}

/**
 * Returns an input object for backward local contrast normalization layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(LayerDataId id) const
{
    layers::LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets input for the backward local contrast normalization layer
 * \param[in] id    Identifier of the input  object
 * \param[in] value Input object to set
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &value)
{
    layers::LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the local contrast normalization layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    if (!algParameter->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Input::check(parameter, method));

    data_management::TensorPtr centeredDataTensor = get(auxCenteredData);

    DAAL_CHECK_STATUS(s, data_management::checkTensor(centeredDataTensor.get(), auxCenteredDataStr()));

    const services::Collection<size_t> &dataDims = centeredDataTensor->getDimensions();
    size_t nDims = dataDims.size();

    if( nDims != 4 ) return services::Status( services::ErrorIncorrectNumberOfDimensionsInTensor );

    services::Collection<size_t> sigmaDims = dataDims;

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

    services::Collection<size_t> cDims = sigmaDims;

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

    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(layers::backward::inputGradient).get(), inputGradientStr(), &dataDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxSigma).get(), auxSigmaStr(), &sigmaDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxC).get(), auxCStr(), &cDims));
    DAAL_CHECK_STATUS(s, data_management::checkTensor(get(auxInvMax).get(), auxInvMaxStr(), &sigmaDims));
    return s;
}

/**
 * Default constructor
 */
Result::Result() : layers::backward::Result() {}

/**
 * Checks the result of the local contrast normalization layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(par);
    if (!algParameter->propagateGradient) { return services::Status(); }

    services::Status s;
    DAAL_CHECK_STATUS(s, layers::backward::Result::check(input, par, method));

    const Input *algInput = static_cast<const Input *>(input);

    return data_management::checkTensor(get(layers::backward::gradient).get(), gradientStr(), &(algInput->get(auxCenteredData)->getDimensions()));
}

}// namespace interface1
}// namespace forward
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
