/* file: locallyconnected2d_layer_backward.cpp */
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
//  Implementation of locally connected 2d calculation algorithm and types methods.
//--
*/

#include "locallyconnected2d_layer_backward_types.h"
#include "locallyconnected2d_layer_types.h"

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
namespace backward
{
namespace interface1
{
/**
 * Default constructor
 */
Input::Input() {};

/**
 * Returns an input object for backward 2D locally connected layer
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
data_management::TensorPtr Input::get(LayerDataId id) const
{
    LayerDataPtr layerData =
        services::staticPointerCast<layers::LayerData, data_management::SerializationIface>(Argument::get(layers::backward::inputFromForward));
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets input for the backward 2D locally connected layer
 * \param[in] id    Identifier of the input  object
 * \param[in] value Input object to set
 */
void Input::set(LayerDataId id, const data_management::TensorPtr &value)
{
    LayerDataPtr layerData = get(layers::backward::inputFromForward);
    (*layerData)[id] = value;
}

/**
 * Checks an input object of the 2D locally connected layer
 * \param[in] parameter %Parameter of layer
 * \param[in] method    Computation method of the layer
 */
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    layers::backward::Input::check(parameter, method);
    if( this->_errors->size() > 0 ) { return; }

    const Parameter *param = static_cast<const Parameter *>(parameter);

    data_management::TensorPtr xTensor = get(auxData);
    if (!data_management::checkTensor(xTensor.get(), this->_errors.get(), auxDataStr())) { return; }

    const services::Collection<size_t> &xDims = xTensor->getDimensions();

    int a = (int)xDims[param->indices.dims[0]] + 2 * param->paddings.size[0] - (int)param->kernelSizes.size[0];
    int b = (int)xDims[param->indices.dims[1]] + 2 * param->paddings.size[1] - (int)param->kernelSizes.size[1];

    DAAL_CHECK(a > 0 || b > 0, ErrorIncorrectParameter);
    DAAL_CHECK_EX(xDims[param->groupDimension] % param->nGroups == 0 || param->nKernels % param->nGroups == 0, ErrorIncorrectParameter,
                  ParameterName, nGroupsStr());

    size_t l3 = (size_t)a / param->strides.size[0] + 1;
    size_t l4 = (size_t)b / param->strides.size[1] + 1;

    services::Collection<size_t> gradDims;
    for(size_t i = 0; i < xDims.size(); i++)
    {
        if(i == param->indices.dims[0]) { gradDims.push_back(l3); }
        else if(i == param->indices.dims[1]) { gradDims.push_back(l4); }
        else if(i == param->groupDimension) { gradDims.push_back(param->nKernels); }
        else { gradDims.push_back( xDims[i] ); }
    }

    services::Collection<size_t> wDims;
    wDims << param->nKernels << l3 << l4 << xDims[param->groupDimension] << param->kernelSizes.size[0] << param->kernelSizes.size[1];

    if (!data_management::checkTensor(get(layers::backward::inputGradient).get(), this->_errors.get(), inputGradientStr(), &gradDims)) { return; }
    if (!data_management::checkTensor(get(auxWeights).get(), this->_errors.get(), auxWeightsStr(), &wDims)) { return; }
}
/**
 * Default constructor
 */
Result::Result() : layers::backward::Result() {}

/**
 * Checks the result of the 2D locally connected layer
 * \param[in] input   %Input object of the layer
 * \param[in] par     %Parameter of the layer
 * \param[in] method  Computation method of the layer
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    layers::backward::Result::check(input, par, method);
    if( this->_errors->size() > 0 ) { return; }

    const Input *algInput = static_cast<const Input *>(input);
    const Parameter *param = static_cast<const Parameter *>(par);

    if (!data_management::checkTensor(get(layers::backward::gradient).get(), this->_errors.get(), gradientStr(),
                                      &(algInput->get(auxData)->getDimensions()))) { return; }
    if (!data_management::checkTensor(get(layers::backward::weightDerivatives).get(), this->_errors.get(), weightDerivativesStr(),
                                      &(algInput->get(auxWeights)->getDimensions()))) { return; }

    services::Collection<size_t> bDims;
    getBiasesDims(algInput, param, bDims);

    if (!data_management::checkTensor(get(layers::backward::biasDerivatives).get(), this->_errors.get(), biasDerivativesStr(), &bDims)) { return; }
}

void Result::getBiasesDims(const Input *algInput, const Parameter *param, services::Collection<size_t> &bDims) const
{
    data_management::TensorPtr auxDataTensor  = algInput->get(auxData);
    const services::Collection<size_t> &xDims = auxDataTensor->getDimensions();

    size_t l3 = (xDims[param->indices.dims[0]] + 2 * param->paddings.size[0] - param->kernelSizes.size[0]) / param->strides.size[0] + 1;
    size_t l4 = (xDims[param->indices.dims[1]] + 2 * param->paddings.size[1] - param->kernelSizes.size[1]) / param->strides.size[1] + 1;

    bDims << param->nKernels << l3 << l4;
}

}// namespace interface1
}// namespace backward
}// namespace locallyconnected2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
