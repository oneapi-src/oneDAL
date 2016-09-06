/* file: tanh_layer_types.h */
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
    auxValue = 2 /*!< Value computed at the forward stage of the layer */
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
struct Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the tanh layer
     */
    Parameter() {};
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
