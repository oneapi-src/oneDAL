/* file: softmax_cross_layer_types.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
    auxProbabilities = 2, /*!< Tensor that stores probabilities for the forward softmax cross-entropy layer */
    auxGroundTruth = 3, /*!< Tensor that stores ground truth data for the forward softmax cross-entropy layer */
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
class Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the softmax cross-entropy layer
    *  \param[in] accuracyThreshold_  Value needed to avoid degenerate cases in logarithm computing
    */
    Parameter(const double accuracyThreshold_ = 1.0e-04) : accuracyThreshold(accuracyThreshold_)
    {};
    double accuracyThreshold; /*!< Value needed to avoid degenerate cases in logarithm computing */
    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const
    {}
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
