/* file: concat_layer.h */
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
//  Implementation of the concat layer
//--
*/

#ifndef __CONCAT_LAYER_H__
#define __CONCAT_LAYER_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/concat/concat_layer_types.h"
#include "algorithms/neural_networks/layers/concat/concat_layer_forward.h"
#include "algorithms/neural_networks/layers/concat/concat_layer_backward.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the concat layer
 */
namespace concat
{
namespace interface1
{
/**
 * @ingroup concat
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__BATCH"></a>
 * \brief Provides methods for the concat layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-CONCATBACKWARD-ALGORITHM">Backward concat layer description and usage models</a> -->
 * <!-- \n<a href="DAAL-REF-CONCATFORWARD-ALGORITHM">Forward concat layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for concat layer, double or float
 * \tparam method           Concat layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method  Computation methods
 *
 * \par References
 *      - \ref backward::interface1::Batch "backward::Batch" class
 *      - \ref forward::interface1::Batch "forward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public LayerIface
{
public:

    Parameter parameter; /*!< Parameters of the algorithm */

    /** \brief Default constructor */

    Batch(size_t concatDimension = 0): parameter(concatDimension)
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

} // namespace concat
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
