/* file: abs_layer_types.h */
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
    auxData = 2 /*!< Data processed at the forward stage of the layer */
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
struct Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the abs layer
     */
    Parameter() {};
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
