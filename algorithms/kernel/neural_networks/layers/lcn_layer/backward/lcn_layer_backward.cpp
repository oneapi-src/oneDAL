/* file: lcn_layer_backward.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
/**
 * Default constructor
 */
Input::Input() {};

/**
 * Returns an input object for backward local contrast normalization layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
services::SharedPtr<data_management::Tensor> Input::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets input for the backward local contrast normalization layer
 * \param[in] id    Identifier of the input  object
 * \param[in] value Input object to set
 */
void Input::set(LayerDataId id, const services::SharedPtr<data_management::Tensor> &value)
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the local contrast normalization layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    layers::backward::Input::check(parameter, method);
    if( this->_errors->size() > 0 ) { return; }

    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    services::SharedPtr<data_management::Tensor> centeredDataTensor = get(auxCenteredData);

    if (!data_management::checkTensor(centeredDataTensor.get(), this->_errors.get(), auxCenteredDataStr())) { return; }

    const services::Collection<size_t> &dataDims = centeredDataTensor->getDimensions();
    size_t nDims = dataDims.size();

    if( nDims != 4 )
    { this->_errors->add( services::ErrorIncorrectNumberOfDimensionsInTensor ); return; }

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

    if (!data_management::checkTensor(get(layers::backward::inputGradient).get(), this->_errors.get(), inputGradientStr(), &dataDims)) { return; }
    if (!data_management::checkTensor(get(auxSigma).get(), this->_errors.get(), auxSigmaStr(), &sigmaDims)) { return; }
    if (!data_management::checkTensor(get(auxC).get(),     this->_errors.get(), auxCStr(), &cDims))         { return; }
    if (!data_management::checkTensor(get(auxInvMax).get(),   this->_errors.get(), auxInvMaxStr(), &sigmaDims))   { return; }
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
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    layers::backward::Result::check(input, par, method);
    if( this->_errors->size() > 0 ) { return; }

    const Input *algInput = static_cast<const Input *>(input);

    if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(),
                                      &(algInput->get(auxCenteredData)->getDimensions()))) { return; }
}

}// namespace interface1
}// namespace forward
}// namespace lcn
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
