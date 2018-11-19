/* file: abs_layer_types.h */
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
//  Implementation of the absolute value (abs) layer interface
//--
*/

#ifndef __ABS_LAYER_TYPES_H__
#define __ABS_LAYER_TYPES_H__

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
 * @defgroup abs_layers Absolute Value (Abs) Layer
 * \copydoc daal::algorithms::neural_networks::layers::abs
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes of the abs layer
 */
namespace abs
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ABS__METHOD"></a>
 * \brief Computation methods for the abs layer
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ABS__LAYERDATAID"></a>
 * \brief Available identifiers of input objects for the abs layer
 */
enum LayerDataId
{
    auxData = layers::lastLayerInputLayout + 1, /*!< Data processed at the forward stage of the layer */
    lastLayerDataId = auxData
};

/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ABS__PARAMETER"></a>
 * \brief Parameters for the abs layer
 *
 * \snippet neural_networks/layers/abs/abs_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the abs layer
     */
    Parameter();
};
/* [Parameter source code] */
} // namespace interface1
using interface1::Parameter;

} // namespace abs
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
