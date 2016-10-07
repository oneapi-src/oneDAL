/* file: concat_layer_types.h */
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
//  Implementation of the concat layer
//--
*/

#ifndef __CONCAT_LAYER_TYPES_H__
#define __CONCAT_LAYER_TYPES_H__

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
 * @defgroup concat Concat Layer
 * \copydoc daal::algorithms::neural_networks::layers::concat
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the concat layer
 */
namespace concat
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__METHOD"></a>
 * Computation methods for the concat layer
 */
enum Method
{
    defaultDense = 0, /*!<  Default: performance-oriented method */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__LAYERDATAID"></a>
 * \brief Identifiers of input objects for the backward concat layer and results for the forward concat layer
 */

enum LayerDataId
{
    auxInputDimensions = 2  /*!< Numeric table of dimensions along which concatenation is implemented*/
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONCAT__PARAMETER"></a>
 * \brief concat layer parameters
 */
class DAAL_EXPORT Parameter: public layers::Parameter
{
public:
    /**
    *  Constructs parameters of the forward concat layer
    *  \param[in] concatDimension   Index of dimension along which concatenation is implemented
    */
    Parameter(size_t concatDimension = 0);

    size_t concatDimension;    /*!< Index of dimension along which concatenation is implemented*/
};

} // namespace interface1
using interface1::Parameter;

} // namespace concat
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal
#endif
