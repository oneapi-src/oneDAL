/* file: neural_networks_training_types.h */
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

#ifndef __NEURAL_NETWORKS_TRAINING_TYPES_H__
#define __NEURAL_NETWORKS_TRAINING_TYPES_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "neural_networks_training_input.h"
#include "neural_networks_training_result.h"
#include "algorithms/neural_networks/layers/layer.h"

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
 * @defgroup neural_networks_training Training
 * \copydoc daal::algorithms::neural_networks::training
 * @ingroup neural_networks
 * @{
 */
/**
* \brief Contains classes for training the model of the neural network
*/
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__METHOD"></a>
 * \brief Computation methods for the neural network model based training
 */
enum Method
{
    defaultDense = 0,     /*!< Default: Feedforward neural network */
    feedforwardDense = 0  /*!< Feedforward neural network */
};

}
}
/** @} */
}
} // namespace daal
#endif
