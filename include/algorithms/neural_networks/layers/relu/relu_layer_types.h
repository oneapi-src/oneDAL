/* file: relu_layer_types.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the rectifier linear unit (relu) layer
//--
*/

#ifndef __RELU_LAYER_TYPES_H__
#define __RELU_LAYER_TYPES_H__

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
 * @defgroup relu_layers Rectifier Linear Unit (ReLU) Layer
 * \copydoc daal::algorithms::neural_networks::layers::relu
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the relu layer
 */
namespace relu
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RELU__METHOD"></a>
 * Computation methods for the relu layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RELU__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward relu layer and results for the forward relu layer
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
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RELU__PARAMETER"></a>
 * \brief Parameters for the relu layer
 *
 * \snippet neural_networks/layers/relu/relu_layer_types.h Parameter source code
 * \DAAL_DEPRECATED
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of the relu layer
     * \DAAL_DEPRECATED
     */
    Parameter();
};
/* [Parameter source code] */
} // namespace interface1
using interface1::Parameter;

} // namespace relu
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
