/* file: pooling3d_layer_types.h */
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
//  Implementation of 3D pooling layer.
//--
*/

#ifndef __POOLING3D_LAYER_TYPES_H__
#define __POOLING3D_LAYER_TYPES_H__

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
 * @defgroup pooling3d Three-dimensional Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::pooling3d
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the three-dimensional (3D) pooling layer
 */
namespace pooling3d
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__KERNELSIZE"></a>
 * \brief Data structure representing the size of the 3D subtensor
 *        from which the element is computed
 */
struct KernelSizes
{
    /**
     * Constructs the structure representing the size of the 3D subtensor
     * from which the element is computed
     * \param[in]  first  Size of the first dimension of the 3D subtensor
     * \param[in]  second Size of the second dimension of the 3D subtensor
     * \param[in]  third  Size of the third dimension of the 3D subtensor
     */
    KernelSizes(size_t first = 2, size_t second = 2, size_t third = 2)
    {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }
    size_t size[3];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__STRIDE"></a>
 * \brief Data structure representing the intervals on which the subtensors for pooling are computed
 */
struct Strides
{
    /**
     * Constructs the structure representing the intervals on which the subtensors for pooling are computed
     * \param[in]  first  Interval over the first dimension on which the pooling is performed
     * \param[in]  second Interval over the second dimension on which the pooling is performed
     * \param[in]  third  Interval over the third dimension on which the pooling is performed
     */
    Strides(size_t first = 2, size_t second = 2, size_t third = 2)
    {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }
    size_t size[3];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__PADDING"></a>
 * \brief Data structure representing the number of data elements to implicitly add
 *        to each size of the three-dimensional subtensor on which pooling is performed
 */
struct Paddings
{
    /**
     * Constructs the structure representing the number of data elements to implicitly add
     * to each size of the three-dimensional subtensor on which pooling is performed
     * \param[in]  first  Number of data elements to add to the the first dimension of the three-dimensional subtensor
     * \param[in]  second Number of data elements to add to the the second dimension of the three-dimensional subtensor
     * \param[in]  third  Number of data elements to add to the the third dimension of the three-dimensional subtensor
     */
    Paddings(size_t first = 2, size_t second = 2, size_t third = 2)
    {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }
    size_t size[3];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__SPATIALDIMENSIONS"></a>
 * \brief Data structure representing the indices of the three dimensions on which pooling is performed
 */
struct Indices
{
    /**
     * Constructs the structure representing the indices of the three dimensions on which pooling is performed
     * \param[in]  first  The first dimension index
     * \param[in]  second The second dimension index
     * \param[in]  third  The third dimension index
     */
    Indices(size_t first = 0, size_t second = 1, size_t third = 2)
    {
        size[0] = first;
        size[1] = second;
        size[2] = third;
    }
    size_t size[3];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__PARAMETER"></a>
 * \brief Parameters for the forward and backward pooling layers
 *
 * \snippet neural_networks/layers/pooling3d/pooling3d_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of 3D pooling layer
     * \param[in] firstIndex        Index of the first of three dimensions on which the pooling is performed
     * \param[in] secondIndex       Index of the second of three dimensions on which the pooling is performed
     * \param[in] thirdIndex        Index of the third of three dimensions on which the pooling is performed
     * \param[in] firstKernelSize   Size of the first dimension of three-dimensional subtensor for which the kernel is applied
     * \param[in] secondKernelSize  Size of the second dimension of three-dimensional subtensor for which the kernel is applied
     * \param[in] thirdKernelSize   Size of the third dimension of three-dimensional subtensor for which the kernel is applied
     * \param[in] firstStride       Interval over the first dimension on which the pooling is performed
     * \param[in] secondStride      Interval over the second dimension on which the pooling is performed
     * \param[in] thirdStride       Interval over the third dimension on which the pooling is performed
     * \param[in] firstPadding      Number of data elements to implicitly add to the the first dimension
     *                              of the three-dimensional subtensor on which the pooling is performed
     * \param[in] secondPadding     Number of data elements to implicitly add to the the second dimension
     *                              of the three-dimensional subtensor on which the pooling is performed
     * \param[in] thirdPadding      Number of data elements to implicitly add to the the third dimension
     *                              of the three-dimensional subtensor on which the pooling is performed
     */
    Parameter(size_t firstIndex, size_t secondIndex, size_t thirdIndex,
    size_t firstKernelSize = 2, size_t secondKernelSize = 2, size_t thirdKernelSize = 2,
              size_t firstStride = 2, size_t secondStride = 2, size_t thirdStride = 2,
              size_t firstPadding = 0, size_t secondPadding = 0, size_t thirdPadding = 0);

    Strides strides;            /*!< Data structure representing the intervals on which the subtensors for pooling are selected */
    Paddings paddings;          /*!< Data structure representing the number of data elements to implicitly add
                                     to each size of the three-dimensional subtensor on which pooling is performed */
    KernelSizes kernelSizes;    /*!< Data structure representing the size of the three-dimensional subtensor
                                     from which the maximum element is selected */
    Indices indices;            /*!< Indices of the three dimensions on which pooling is performed */
};
/* [Parameter source code] */

} // interface1
using interface1::Parameter;
using interface1::KernelSizes;
using interface1::Strides;
using interface1::Paddings;
using interface1::Indices;

} // namespace pooling3d
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
