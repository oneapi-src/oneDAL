/* file: loss_layer.h */
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
//  Implementation of the loss layer.
//--
*/

#ifndef __LOSS_LAYER_H__
#define __LOSS_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_forward.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_backward.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * @defgroup loss Loss Layer
 * \copydoc daal::algorithms::neural_networks::layers::loss
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for loss layer
 */
namespace loss
{
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__BATCH"></a>
 * \brief Provides methods for the loss layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOSSFORWARD-ALGORITHM">Forward loss layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-LOSSBACKWARD-ALGORITHM">Backward loss layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the loss layer, double or float
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref forward::interface1::Batch "forward::Batch" class
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
class Batch : public LayerIface
{
public:
    /** Default constructor */
    Batch()
    {}

};
} // namespace interface1
using interface1::Batch;

} // namespace loss
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
