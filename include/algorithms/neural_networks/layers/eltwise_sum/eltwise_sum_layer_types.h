/* file: eltwise_sum_layer_types.h */
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
//  Element-wise sum layer parameter structure.
//--
*/

#ifndef __ELTWISE_SUM_LAYER_TYPES_H__
#define __ELTWISE_SUM_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * @defgroup eltwise_sum Element-wise Sum Layer
 * \copydoc daal::algorithms::neural_networks::layers::eltwise_sum
 * @ingroup layers
 * @{
 */
namespace eltwise_sum
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__METHOD"></a>
 * Available methods to compute forward and backward element-wise sum layer
 */
enum Method
{
    defaultDense = 0  /*!< Default: performance-oriented method. */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__LAYERDATAID"></a>
 * \brief Available identifiers of input tensors for the backward  element-wise sum layer
 *        and identifiers of result tensors for the forward element-wise sum layer
 */
enum LayerDataId
{
    auxCoefficients,  /*!< Coefficients obtained from the forward stage of the layer */
    lastLayerDataId = auxCoefficients
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__LAYERDATANUMERICTABLEID"></a>
 * \brief Available identifiers of input numeric tables for the backward  element-wise sum layer
 *        and identifiers of result numeric tables for the forward element-wise sum layer
 */
enum LayerDataNumericTableId
{
    auxNumberOfCoefficients     = lastLayerDataId + 1,  /*!< Numeric table of size 1 x 1 that contains the number of coefficients */
    lastLayerDataNumericTableId = auxNumberOfCoefficients
};


namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ELTWISE_SUM__PARAMETER"></a>
 * \brief Parameters for the element-wise sum layer
 *
 * \snippet neural_networks/layers/eltwise_sum/eltwise_sum_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the element-wise sum layer
     */
    Parameter();
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Parameter;

} // namespace eltwise_sum
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
