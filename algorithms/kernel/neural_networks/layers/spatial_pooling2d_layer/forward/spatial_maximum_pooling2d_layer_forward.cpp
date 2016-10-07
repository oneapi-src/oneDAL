/* file: spatial_maximum_pooling2d_layer_forward.cpp */
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
//  Implementation of spatial pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_maximum_pooling2d_layer_types.h"
#include "spatial_maximum_pooling2d_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_maximum_pooling2d
{
namespace forward
{
namespace interface1
{
/** Default constructor */
Result::Result() {}

/**
 * Returns the result of the forward spatial pyramid maximum 2D pooling layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
data_management::TensorPtr Result::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
    return services::staticPointerCast<data_management::Tensor, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Returns the result of the forward spatial pyramid maximum 2D pooling layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(LayerDataNumericTableId id) const
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of the forward spatial pyramid maximum 2D pooling layer
 * \param[in] id Identifier of the result
 * \param[in] ptr Result
 */
void Result::set(LayerDataId id, const data_management::TensorPtr &ptr)
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
    (*layerData)[id] = ptr;
}

/**
 * Sets the result of the forward spatial pyramid maximum 2D pooling layer
 * \param[in] id Identifier of the result
 * \param[in] ptr Result
 */
void Result::set(LayerDataNumericTableId id, const data_management::NumericTablePtr &ptr)
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
    (*layerData)[id] = ptr;
}

/**
 * Checks the result of the forward spatial pyramid maximum 2D pooling layer
 * \param[in] input     %Input of the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method of the layer
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    spatial_pooling2d::forward::Result::check(input, parameter, method);
}

}// namespace interface1
}// namespace forward
}// namespace spatial_maximum_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
