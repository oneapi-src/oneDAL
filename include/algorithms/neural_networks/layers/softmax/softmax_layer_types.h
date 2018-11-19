/* file: softmax_layer_types.h */
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
//  Implementation of the softmax layer interface.
//--
*/

#ifndef __SOFTMAX_LAYER_TYPES_H__
#define __SOFTMAX_LAYER_TYPES_H__

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
 * @defgroup softmax_layers Softmax Layer
 * \copydoc daal::algorithms::neural_networks::layers::softmax
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes of the softmax layer
 */
namespace softmax
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__METHOD"></a>
 * \brief Computation methods for the softmax layer
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__LAYERDATAID"></a>
 * \brief Available identifiers of input objects for the softmax layer
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SOFTMAX__PARAMETER"></a>
 * \brief Parameters for the softmax layer
 *
 * \snippet neural_networks/layers/softmax/softmax_layer_types.h Parameter source code
 *
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
     *  Constructs parameters of the softmax layer
     *  \param[in] _dimension   Dimension index to calculate softmax
     */
    Parameter(size_t _dimension = 1);

    /**
     *  Constructs parameters of the softmax layer by copying another parameters of the softmax layer
     *  \param[in] other    Parameters of the softmax layer
     */
    Parameter(const Parameter &other);

    size_t dimension; /*!< Dimension index to calculate softmax */
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Parameter;

} // namespace softmax
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
