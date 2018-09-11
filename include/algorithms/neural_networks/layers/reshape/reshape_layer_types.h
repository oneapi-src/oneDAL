/* file: reshape_layer_types.h */
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

#ifndef __RESHAPE_LAYER_TYPES_H__
#define __RESHAPE_LAYER_TYPES_H__

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
 * @defgroup reshape_layers Reshape Layer
 * \copydoc daal::algorithms::neural_networks::layers::reshape
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes of the reshape layer
 */
namespace reshape
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RESHAPE__UNDEFINEDDIMENSIONSIZE"></a>
 * \brief Constant value used to show that dimension size is calculated in a special way
 */
const size_t undefinedDimensionSize = (size_t)(-1);

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RESHAPE__METHOD"></a>
 * \brief Computation methods for the reshape layer
 */
enum Method
{
    defaultDense = 0 /*!< Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RESHAPE__LAYERDATAID"></a>
 * \brief Available identifiers of input objects for the reshape layer
 */
enum LayerDataId
{
    auxInputDimensions = layers::lastLayerInputLayout + 1,  /*!< Numeric table of dimensions of original input data */
    lastLayerDataId = auxInputDimensions
};

/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RESHAPE__PARAMETER"></a>
 * \brief Parameters for the reshape layer
 *
 * \snippet neural_networks/layers/reshape/reshape_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the reshape layer
     */
    Parameter();

    services::Collection<size_t> reshapeDimensions;
};
/* [Parameter source code] */
} // namespace interface1
using interface1::Parameter;

} // namespace reshape
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
