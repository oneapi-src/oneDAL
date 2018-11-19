/* file: pooling2d_layer_types.h */
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
//  Implementation of 2D pooling layer.
//--
*/

#ifndef __POOLING2D_LAYER_TYPES_H__
#define __POOLING2D_LAYER_TYPES_H__

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
 * @defgroup pooling2d Two-dimensional Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::pooling2d
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the two-dimensional (2D) pooling layer
 */
namespace pooling2d
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__KERNELSIZE"></a>
 * \brief Data structure representing the size of the 2D subtensor
 *        from which the element is computed
 */
struct KernelSizes
{
    /**
     * Constructs the structure representing the size of the 2D subtensor
     * from which the element is computed
     * \param[in]  first  Size of the first dimension of the 2D subtensor
     * \param[in]  second Size of the second dimension of the 2D subtensor
     */
    KernelSizes(size_t first = 2, size_t second = 2) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__STRIDE"></a>
 * \brief Data structure representing the intervals on which the subtensors for pooling are computed
 */
struct Strides
{
    /**
     * Constructs the structure representing the intervals on which the subtensors for pooling are computed
     * \param[in]  first  Interval over the first dimension on which the pooling is performed
     * \param[in]  second Interval over the second dimension on which the pooling is performed
     */
    Strides(size_t first = 2, size_t second = 2) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__PADDING"></a>
 * \brief Data structure representing the number of data elements to implicitly add
 *        to each side of the 2D subtensor on which pooling is performed
 */
struct Paddings
{
    /**
     * Constructs the structure representing the number of data elements to implicitly add
     * to each side of the 2D subtensor on which pooling is performed
     * \param[in]  first  Number of data elements to add to the the first dimension of the 2D subtensor
     * \param[in]  second Number of data elements to add to the the second dimension of the 2D subtensor
     */
    Paddings(size_t first = 2, size_t second = 2) { size[0] = first; size[1] = second; }
    size_t size[2];
};

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
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING2D__PARAMETER"></a>
 * \brief Parameters for the forward and backward two-dimensional pooling layers
 *
 * \snippet neural_networks/layers/pooling2d/pooling2d_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of 2D pooling layer
     * \param[in] firstIndex        Index of the first of two dimensions on which the pooling is performed
     * \param[in] secondIndex       Index of the second of two dimensions on which the pooling is performed
     * \param[in] firstKernelSize   Size of the first dimension of 2D subtensor for which the element is computed
     * \param[in] secondKernelSize  Size of the second dimension of 2D subtensor for which the element is computed
     * \param[in] firstStride       Interval over the first dimension on which the pooling is performed
     * \param[in] secondStride      Interval over the second dimension on which the pooling is performed
     * \param[in] firstPadding      Number of data elements to implicitly add to the the first dimension
     *                              of the 2D subtensor on which the pooling is performed
     * \param[in] secondPadding     Number of data elements to implicitly add to the the second dimension
     *                              of the 2D subtensor on which the pooling is performed
     */
    Parameter(size_t firstIndex, size_t secondIndex, size_t firstKernelSize = 2, size_t secondKernelSize = 2,
              size_t firstStride = 2, size_t secondStride = 2, size_t firstPadding = 0, size_t secondPadding = 0);

    Strides strides;            /*!< Data structure representing the intervals on which the subtensors for pooling are computed */
    Paddings paddings;          /*!< Data structure representing the number of data elements to implicitly add
                                     to each size of the 2D subtensor on which pooling is performed */
    KernelSizes kernelSizes;    /*!< Data structure representing the size of the 2D subtensor
                                     from which the element is computed */
    Indices indices;            /*!< Indices of the two dimensions on which pooling is performed */
};
/* [Parameter source code] */

} // interface1
using interface1::Parameter;
using interface1::KernelSizes;
using interface1::Strides;
using interface1::Paddings;
using interface1::Indices;

} // namespace pooling2d
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
