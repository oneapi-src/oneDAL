/* file: neural_networks_types.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup training_and_prediction
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
typedef services::SharedPtr<ForwardLayers> ForwardLayersPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__BACKWARDLAYERS"></a>
 * \brief Represents a collection of forward stages of neural network layers
 */
typedef services::Collection<layers::backward::LayerIfacePtr > BackwardLayers;
typedef services::SharedPtr<BackwardLayers> BackwardLayersPtr;

} // namespace interface1
using interface1::ForwardLayers;
using interface1::ForwardLayersPtr;
using interface1::BackwardLayers;
using interface1::BackwardLayersPtr;

}
/** @} */
}
} // namespace daal
#endif
