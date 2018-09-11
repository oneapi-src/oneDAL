/* file: tanh_layer_types.h */
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
//  Implementation of the hyperbolic tangent layer.
//--
*/

#ifndef __TANH_LAYER_TYPES_H__
#define __TANH_LAYER_TYPES_H__

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
 * @defgroup tanh_layers Hyperbolic Tangent Layer
 * \copydoc daal::algorithms::neural_networks::layers::tanh
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the hyperbolic tangent layer
 */
namespace tanh
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__METHOD"></a>
 * Computation methods for the hyperbolic tangent layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward hyperbolic tangent layer and results for the forward hyperbolic tangent layer
 */
enum LayerDataId
{
    auxValue = layers::lastLayerInputLayout + 1, /*!< Value computed at the forward stage of the layer */
    lastLayerDataId = auxValue
};

/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__PARAMETER"></a>
 * \brief Parameters for the tanh layer
 *
 * \snippet neural_networks/layers/tanh/tanh_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the tanh layer
     */
    Parameter();
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Parameter;

} // namespace tanh
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
