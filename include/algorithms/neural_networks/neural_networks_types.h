/* file: neural_networks_types.h */
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
//  Implementation of neural network algorithm interface.
//--
*/

#ifndef __NEURAL_NETWORKS_TYPES_H__
#define __NEURAL_NETWORKS_TYPES_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward.h"
#include "algorithms/neural_networks/layers/layer_backward.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup neural_networks Neural Networks
 * \copydoc daal::algorithms::neural_networks
 * @ingroup classification
 * @{
 */
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__FORWARDLAYERS"></a>
 * \brief Represents a collection of forward stages of neural network layers
 */
typedef services::Collection<layers::forward::LayerIfacePtr > ForwardLayers;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__BACKWARDLAYERS"></a>
 * \brief Represents a collection of forward stages of neural network layers
 */
typedef services::Collection<layers::backward::LayerIfacePtr > BackwardLayers;

} // namespace interface1
using interface1::ForwardLayers;
using interface1::BackwardLayers;

}
/** @} */
}
} // namespace daal
#endif
