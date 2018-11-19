/* file: softmax_cross_layer_types.h */
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
//  Implementation of the softmax cross-entropy layer types.
//--
*/

#ifndef __NEURAL_NENTWORK_LOSS_SOFTMAX_CROSS_LAYER_TYPES_H__
#define __NEURAL_NENTWORK_LOSS_SOFTMAX_CROSS_LAYER_TYPES_H__

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
 * @defgroup softmax_cross Softmax Cross-entropy Layer
 * \copydoc daal::algorithms::neural_networks::layers::loss::softmax_cross
 * @ingroup loss
 * @{
 */
namespace softmax_cross
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__SOFTMAX_CROSS__METHOD"></a>
 * \brief Computation methods for the softmax cross-entropy layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__SOFTMAX_CROSS__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward softmax cross-entropy layer and results for the forward softmax cross-entropy layer
 */
enum LayerDataId
{
    auxProbabilities = layers::lastLayerInputLayout + 1, /*!< Tensor that stores probabilities for the forward softmax cross-entropy layer */
    auxGroundTruth, /*!< Tensor that stores ground truth data for the forward softmax cross-entropy layer */
    lastLayerDataId = auxGroundTruth
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__SOFTMAX_CROSS__PARAMETER"></a>
 * \brief Parameters for the softmax cross-entropy layer
 *
 * \snippet neural_networks/layers/loss/softmax_cross_layer_types.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the softmax cross-entropy layer
    *  \param[in] accuracyThreshold_  Value needed to avoid degenerate cases in logarithm computing
    *  \param[in] dimension_          Dimension index to calculate softmax cross-entropy
    */
    Parameter(const double accuracyThreshold_ = 1.0e-04, const size_t dimension_ = 1);

    double accuracyThreshold; /*!< Value needed to avoid degenerate cases in logarithm computing */
    size_t dimension;         /*!< Dimension index to calculate softmax cross-entropy */
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

} // namespace softmax_cross
/** @} */
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
