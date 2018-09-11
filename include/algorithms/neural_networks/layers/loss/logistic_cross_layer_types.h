/* file: logistic_cross_layer_types.h */
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
//  Implementation of the logistic cross-entropy layer types.
//--
*/

#ifndef __NEURAL_NENTWORK_LOSS_LOGISTIC_CROSS_LAYER_TYPES_H__
#define __NEURAL_NENTWORK_LOSS_LOGISTIC_CROSS_LAYER_TYPES_H__

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
namespace loss
{
/**
 * @defgroup logistic_cross Logistic Cross-entropy Layer
 * \copydoc daal::algorithms::neural_networks::layers::loss::logistic_cross
 * @ingroup loss
 * @{
 */
namespace logistic_cross
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOGISTIC_CROSS__METHOD"></a>
 * \brief Computation methods for the logistic cross-entropy layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOGISTIC_CROSS__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward logistic cross-entropy layer and results for the forward logistic cross-entropy layer
 */
enum LayerDataId
{
    auxData = layers::lastLayerInputLayout + 1, /*!< Data processed at the forward stage of the layer */
    auxGroundTruth, /*!< Tensor that stores ground truth data for the forward logistic cross-entropy layer */
    lastLayerDataId = auxGroundTruth
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOGISTIC_CROSS__PARAMETER"></a>
 * \brief Parameters for the logistic cross-entropy layer
 *
 * \snippet neural_networks/layers/loss/logistic_cross_layer_types.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the logistic cross-entropy layer
    */
    Parameter();
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

} // namespace logistic_cross
/** @} */
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
