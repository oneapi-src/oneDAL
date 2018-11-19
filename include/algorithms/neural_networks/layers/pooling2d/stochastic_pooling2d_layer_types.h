/* file: stochastic_pooling2d_layer_types.h */
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
//  Implementation of stochastic 2D pooling layer.
//--
*/

#ifndef __STOCHASTIC_POOLING2D_LAYER_TYPES_H__
#define __STOCHASTIC_POOLING2D_LAYER_TYPES_H__

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
 * @defgroup stochastic_pooling2d Two-dimensional Stochastic Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::stochastic_pooling2d
 * @ingroup pooling2d
 * @{
 */
namespace stochastic_pooling2d
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__STOCHASTIC_POOLING2D__METHOD"></a>
 * \brief Computation methods for the stochastic 2D pooling layer
 */
enum Method
{
    defaultDense = 0    /*!< Default: performance-oriented method */
};

/**
 * \brief Identifiers of input tensors for the backward stochastic 2D pooling layer
 *        and results for the forward stochastic 2D pooling layer
 */
enum LayerDataId
{
    auxSelectedIndices,         /*!< p-dimensional tensor that stores the positions of stochastic elements */
    lastLayerDataId = auxSelectedIndices
};

/**
 * \brief Identifiers of input numeric tables for the backward stochastic 2D pooling layer
 *        and results for the forward stochastic 2D pooling layer
 */
enum LayerDataNumericTableId
{
    auxInputDimensions  = lastLayerDataId + 1,         /*!< Numeric table of size 1 x p that contains the sizes
                                          of the dimensions of the input data tensor */
    lastLayerDataNumericTableId = auxInputDimensions
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__STOCHASTIC_POOLING2D__PARAMETER"></a>
 * \brief Parameters for the stochastic 2D pooling layer
 *
 * \snippet neural_networks/layers/pooling2d/stochastic_pooling2d_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public pooling2d::Parameter
{
    /**
     * Constructs the parameters of 2D pooling layer
     * \param[in] firstIndex        Index of the first of two dimensions on which the pooling is performed
     * \param[in] secondIndex       Index of the second of two dimensions on which the pooling is performed
     * \param[in] firstKernelSize   Size of the first dimension of 2D subtensor for which the stochastic element is selected
     * \param[in] secondKernelSize  Size of the second dimension of 2D subtensor for which the stochastic element is selected
     * \param[in] firstStride       Interval over the first dimension on which the pooling is performed
     * \param[in] secondStride      Interval over the second dimension on which the pooling is performed
     * \param[in] firstPadding      Number of data elements to implicitly add to the the first dimension
     *                              of the 2D subtensor on which the pooling is performed
     * \param[in] secondPadding     Number of data elements to implicitly add to the the second dimension
     *                              of the 2D subtensor on which the pooling is performed
     */
    Parameter(size_t firstIndex, size_t secondIndex, size_t firstKernelSize = 2, size_t secondKernelSize = 2,
              size_t firstStride = 2, size_t secondStride = 2, size_t firstPadding = 0, size_t secondPadding = 0);

    size_t seed;                /*!< Seed for multinomial distribution random number generator. \DAAL_DEPRECATED_USE{ engine } */
    engines::EnginePtr engine;  /*!< Engine for multinomial distribution random number generator. */
};
/* [Parameter source code] */

} // interface1
using interface1::Parameter;

} // namespace stochastic_pooling2d
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
