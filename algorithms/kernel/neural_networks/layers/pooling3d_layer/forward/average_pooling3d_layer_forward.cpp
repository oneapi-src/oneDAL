/* file: average_pooling3d_layer_forward.cpp */
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
//  Implementation of average_pooling3d calculation algorithm and types methods.
//--
*/

#include "average_pooling3d_layer_forward_types.h"
#include "average_pooling3d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling3d
{
namespace forward
{
namespace interface1
{
/**
 * Default constructor
 */
Result::Result() {}

/**
 * Returns the result of the forward average 3D pooling layer
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
data_management::NumericTablePtr Result::get(LayerDataId id) const
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
    return services::staticPointerCast<data_management::NumericTable, data_management::SerializationIface>((*layerData)[id]);
}

/**
 * Sets the result of the forward average 3D pooling layer
 * \param[in] id      Identifier of the result
 * \param[in] ptr     Result
 */
void Result::set(LayerDataId id, const data_management::NumericTablePtr &ptr)
{
    services::SharedPtr<layers::LayerData> layerData = get(layers::forward::resultForBackward);
    (*layerData)[id] = ptr;
}

/**
 * Checks the result of the forward average 3D pooling layer
 * \param[in] input     %Input of the layer
 * \param[in] parameter %Parameter of the layer
 * \param[in] method    Computation method of the layer
 */
void Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    pooling3d::forward::Result::check(input, parameter, method);
    if(this->_errors->size() > 0) { return; }
    const Parameter *param =  static_cast<const Parameter *>(parameter);

    if(!param->predictionStage)
    {
        const Input *in = static_cast<const Input *>(input);

        const services::Collection<size_t> &dataDims = in->get(layers::forward::data)->getDimensions();

        data_management::NumericTablePtr auxInputDimensions = get(average_pooling3d::auxInputDimensions);
        if(!data_management::checkNumericTable(auxInputDimensions.get(), this->_errors.get(), auxInputDimensionsStr(), data_management::packed_mask, 0,
                                               dataDims.size(), 1)) { return; }
    }
}

}// namespace interface1
}// namespace forward
}// namespace average_pooling3d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
