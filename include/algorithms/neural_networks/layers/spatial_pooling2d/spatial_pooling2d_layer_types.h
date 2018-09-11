/* file: spatial_pooling2d_layer_types.h */
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
//  Implementation of 2D spatial layer.
//--
*/

#ifndef __SPATIAL_POOLING2D_LAYER_TYPES_H__
#define __SPATIAL_POOLING2D_LAYER_TYPES_H__

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
 * @defgroup spatial_pooling2d Two-dimensional Spatial Pyramid Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::spatial_pooling2d
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the two-dimensional (2D) spatial layer
 */
namespace spatial_pooling2d
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__SPATIALDIMENSIONS"></a>
 * \brief Data structure representing the indices of the two dimensions on which pooling is performed
 */
struct Indices
{
    /**
     * Constructs the structure representing the indices of the two dimensions on which pooling is performed
     * \param[in]  first  The first dimension index
     * \param[in]  second The second dimension index
     */
    Indices(size_t first = 2, size_t second = 3) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_POOLING2D__PARAMETER"></a>
 * \brief Parameters for the forward and backward two-dimensional spatial layers
 *
 * \snippet neural_networks/layers/spatial_pooling2d/spatial_pooling2d_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of 2D spatial layer
     * \param[in] _pyramidHeight     The value of pyramid height
     * \param[in] firstIndex         Index of the first of two dimensions on which the spatial is performed
     * \param[in] secondIndex        Index of the second of two dimensions on which the spatial is performed
     */
    Parameter(size_t _pyramidHeight, size_t firstIndex, size_t secondIndex);

    size_t pyramidHeight;       /*!< The value of pyramid height */
    Indices indices;            /*!< Indices of the two dimensions on which spatial is performed */
};
/* [Parameter source code] */

} // interface1
using interface1::Parameter;
using interface1::Indices;

} // namespace spatial_pooling2d
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
