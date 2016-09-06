/* file: average_pooling2d_layer_types.h */
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
//  Implementation of two-dimensional average pooling layer.
//--
*/

#ifndef __AVERAGE_POOLING2D_LAYER_TYPES_H__
#define __AVERAGE_POOLING2D_LAYER_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/pooling2d/pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * @defgroup average_pooling2d Two-dimensional Average Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::average_pooling2d
 * @ingroup pooling2d
 * @{
 */
namespace average_pooling2d
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING2D__METHOD"></a>
 * \brief Computation methods for the average 2D pooling layer
 */
enum Method
{
    defaultDense = 0  /*!<  Default: performance-oriented method */
};

/**
 * \brief Identifiers of input objects for the backward average 2D pooling layer
 *        and results for the forward average 2D pooling layer
 */
enum LayerDataId
{
    auxInputDimensions = 0  /*!< Numeric table that stores forward average pooling layer results */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING2D__PARAMETER"></a>
 * \brief Parameters for the average 2D pooling layer
 *
 * \snippet neural_networks/layers/pooling2d/average_pooling2d_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter: public pooling2d::Parameter
{
    /**
     * Constructs the parameters of average 2D pooling layer
     * \param[in] firstIndex        Index of the first of two dimensions on which the pooling is performed
     * \param[in] secondIndex       Index of the second of two dimensions on which the pooling is performed
     * \param[in] firstKernelSize   Size of the first dimension of 2D subtensor for which the average element is computed
     * \param[in] secondKernelSize  Size of the second dimension of 2D subtensor for which the average element is computed
     * \param[in] firstStride       Interval over the first dimension on which the pooling is performed
     * \param[in] secondStride      Interval over the second dimension on which the pooling is performed
     * \param[in] firstPadding      Number of data elements to implicitly add to the the first dimension
     *                              of the 2D subtensor on which the pooling is performed
     * \param[in] secondPadding     Number of data elements to implicitly add to the the second dimension
     *                              of the 2D subtensor on which the pooling is performed
     */
    Parameter(size_t firstIndex, size_t secondIndex, size_t firstKernelSize = 2, size_t secondKernelSize = 2,
              size_t firstStride = 2, size_t secondStride = 2, size_t firstPadding = 0, size_t secondPadding = 0) :
        layers::pooling2d::Parameter(firstIndex, secondIndex, firstKernelSize, secondKernelSize,
                                     firstStride, secondStride, firstPadding, secondPadding)
    {}
};
/* [Parameter source code] */

} // interface1
using interface1::Parameter;

} // namespace average_pooling2d
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
