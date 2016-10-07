/* file: layer.h */
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
//  Implementation of neural network layer.
//--
*/

#ifndef __LAYER_H__
#define __LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_types.h"
#include "algorithms/neural_networks/layers/layer_forward.h"
#include "algorithms/neural_networks/layers/layer_backward.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_forward.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * \brief Contains classes for neural network layers
 */
namespace layers
{
namespace interface1
{
/**
 * @ingroup layers
 * @{
 */
/**
* <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LAYERIFACE"></a>
* \brief Abstract class that specifies the interface of layer
*/
class LayerIface: public Base
{
public:
    forward::LayerIfacePtr forwardLayer;   /*!< Forward stage of the layer algorithm */
    backward::LayerIfacePtr backwardLayer; /*!< Backward stage of the layer algorithm */
};
/** }@ */
typedef services::SharedPtr<LayerIface> LayerIfacePtr;
} // interface1
using interface1::LayerIface;
using interface1::LayerIfacePtr;

} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
