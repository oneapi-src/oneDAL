/* file: lcn_layer.h */
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
//  Implementation of the local contrast normalization layer.
//--
*/

#ifndef __NEURAL_NETWORK_LCN_LAYER_H__
#define __NEURAL_NETWORK_LCN_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/lcn/lcn_layer_types.h"
#include "algorithms/neural_networks/layers/lcn/lcn_layer_forward.h"
#include "algorithms/neural_networks/layers/lcn/lcn_layer_backward.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for neural network local contrast normalization layer
 */
namespace lcn
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup lcn_layers
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__BATCH"></a>
 * \brief Computes the result of the forward and backward local contrast normalization layer of neural network in the batch processing mode
 *
 * \tparam algorithmFPType Data type to use in intermediate computations for the local contrast normalization layer, double or float
 * \tparam method          %Batch local contrast normalization layer computation method, \ref Method
 *
 * \par References
 * <!--     - <a href="DAAL-REF-LCNFORWARD-ALGORITHM">Forward local contrast normalization layer description and usage models</a> -->
 *      - \ref forward::interface1::Batch  "forward::Batch" class
 * <!--     - <a href="DAAL-REF-LCNBACKWARD-ALGORITHM">Backward local contrast normalization layer description and usage models</a> -->
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public LayerIface
{
public:
    Parameter parameter; /*!< %Parameters of the layer */

    /** Default constructor */
    Batch()
    {
        forward::Batch<algorithmFPType, method> *forwardLayerObject = new forward::Batch<algorithmFPType, method>(parameter);
        backward::Batch<algorithmFPType, method> *backwardLayerObject = new backward::Batch<algorithmFPType, method>(parameter);

        LayerIface::forwardLayer = services::SharedPtr<forward::Batch<algorithmFPType, method> >(forwardLayerObject);
        LayerIface::backwardLayer = services::SharedPtr<backward::Batch<algorithmFPType, method> >(backwardLayerObject);
    };
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace lcn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
