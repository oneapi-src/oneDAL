/* file: logistic_cross_layer.h */
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
//  Implementation of the logistic cross layer.
//--
*/

#ifndef __NEURAL_NENTWORK_LOSS_LOGISTIC_CROSS_LAYER_H__
#define __NEURAL_NENTWORK_LOSS_LOGISTIC_CROSS_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/loss/loss_layer.h"
#include "algorithms/neural_networks/layers/loss/logistic_cross_layer_types.h"
#include "algorithms/neural_networks/layers/loss/logistic_cross_layer_forward.h"
#include "algorithms/neural_networks/layers/loss/logistic_cross_layer_backward.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace loss
{
/**
 * \brief Contains classes for logistic cross-entropy layer
 */
namespace logistic_cross
{
namespace interface1
{
/**
 * @ingroup logistic_cross
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOGISTIC_CROSS__BATCH"></a>
 * \brief Provides methods for the logistic cross-entropy layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-LOGISTIC_CROSSFORWARD-ALGORITHM">Forward logistic cross-entropy layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-LOGISTIC_CROSSBACKWARD-ALGORITHM">Backward logistic cross-entropy layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the logistic cross-entropy layer, double or float
 * \tparam method           Dropout layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref forward::interface1::Batch "forward::Batch" class
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public loss::Batch
{
public:
    /** Default constructor */
    Batch()
    {
        forward::Batch<algorithmFPType, method> *forwardLayerObject = new forward::Batch<algorithmFPType, method>(parameter);
        backward::Batch<algorithmFPType, method> *backwardLayerObject = new backward::Batch<algorithmFPType, method>(parameter);

        LayerIface::forwardLayer = services::SharedPtr<forward::Batch<algorithmFPType, method> >(forwardLayerObject);
        LayerIface::backwardLayer = services::SharedPtr<backward::Batch<algorithmFPType, method> >(backwardLayerObject);
    };

    Parameter parameter; /*!< Loss layer parameters */
};
/** @} */
} // namespace interface1
using interface1::Batch;

} // namespace logistic_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
