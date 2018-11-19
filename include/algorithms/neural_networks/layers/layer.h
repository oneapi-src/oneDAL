/* file: layer.h */
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
