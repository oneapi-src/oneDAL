/* file: prelu_layer_types.h */
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
//  Implementation of the parametric rectifier linear unit (prelu) layer
//--
*/

#ifndef __PRELU_LAYER_TYPES_H__
#define __PRELU_LAYER_TYPES_H__

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
 * @defgroup prelu Parametric Rectifier Linear Unit (pReLU) Layer
 * \copydoc daal::algorithms::neural_networks::layers::prelu
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the prelu layer
 */
namespace prelu
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__METHOD"></a>
 * Computation methods for the prelu layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward prelu layer and results for the forward prelu layer
 */

enum LayerDataId
{
    auxData = layers::lastLayerInputLayout + 1,       /*!< Data processed at the forward stage of the layer */
    auxWeights,                                                         /*!< Weights of the prelu layer */
    lastLayerDataId = auxWeights
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__PARAMETER"></a>
 * \brief Parameters for the prelu layer
 *
 * \snippet neural_networks/layers/prelu/prelu_layer_types.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the prelu layer
    *  \param[in] _dataDimension    Starting data dimension index to apply weight
    *  \param[in] _weightsDimension Number of weight dimensions
    */
    Parameter(const size_t _dataDimension = 0, const size_t _weightsDimension = 1);

    size_t dataDimension;    /*!< Starting data dimension index to apply weight */
    size_t weightsDimension; /*!< Number of weight dimensions */

    /**
     * Checks the correctness of the parameter
     *
     * \return Status of computations
     */
    virtual services::Status check() const;
};
/* [Parameter source code] */

} // namespace interface1
using interface1::Parameter;

} // namespace prelu
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
