/* file: pooling1d_layer_types.h */
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
//  Implementation of 1D pooling layer.
//--
*/

#ifndef __POOLING1D_LAYER_TYPES_H__
#define __POOLING1D_LAYER_TYPES_H__

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
 * @defgroup pooling1d One-dimensional Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::pooling1d
 * @ingroup layers
 * @{
 */
/**
 * \brief Contains classes for the one-dimensional (1D) pooling layer
 */
namespace pooling1d
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__KERNELSIZE"></a>
 * \brief Data structure representing the size of the 1D subtensor
 *        from which the element is computed
 */
struct KernelSize
{
    /**
     * Constructs the structure representing the size of the 1D subtensor
     * from which the element is computed
     * \param[in]  first  Size of the first dimension of the 1D subtensor
     */
    KernelSize(size_t first = 2) { size[0] = first;}
    size_t size[1];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__STRIDE"></a>
 * \brief Data structure representing the intervals on which the subtensors for pooling are computed
 */
struct Stride
{
    /**
     * Constructs the structure representing the intervals on which the subtensors for pooling are computed
     * \param[in]  first  Interval over the first dimension on which the pooling is performed
     */
    Stride(size_t first = 2) { size[0] = first;}
    size_t size[1];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__PADDING"></a>
 * \brief Data structure representing the number of data elements to implicitly add
 *        to each side of the 1D subtensor on which pooling is performed
 */
struct Padding
{
    /**
     * Constructs the structure representing the number of data elements to implicitly add
     * to each side of the 1D subtensor on which pooling is performed
     * \param[in]  first  Number of data elements to add to the the first dimension of the 1D subtensor
     */
    Padding(size_t first = 0) { size[0] = first;}
    size_t size[1];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__SPATIALDIMENSIONS"></a>
 * \brief Data structure representing the indices of the dimension on which pooling is performed
 */
struct Index
{
    /**
     * Constructs the structure representing the indices of the dimension on which pooling is performed
     * \param[in]  first  The first dimension index
     */
    Index(size_t first = 2) { size[0] = first;}
    size_t size[1];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING1D__PARAMETER"></a>
 * \brief Parameters for the forward and backward pooling layers
 *
 * \snippet neural_networks/layers/pooling1d/pooling1d_layer_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of 1D pooling layer
     * \param[in] index        Index of the dimension on which pooling is performed
     * \param[in] kernelSize   Size of 1D subtensor for which the element is computed
     * \param[in] stride       Interval over the dimension on which the pooling is performed
     * \param[in] padding      Number of data elements to implicitly add to the the dimension
     *                         of the 1D subtensor on which the pooling is performed
     */
    Parameter(size_t index, size_t kernelSize = 2, size_t stride = 2, size_t padding = 0);

    Stride stride;              /*!< Data structure representing the intervals on which the subtensors for pooling are computed */
    Padding padding;            /*!< Data structure representing the number of data elements to implicitly add
                                     to each size of the 1D subtensor on which pooling is performed */
    KernelSize kernelSize;      /*!< Data structure representing the size of the 1D subtensor
                                     from which the element is computed */
    Index index;                /*!< Index of the dimension on which pooling is performed */
};
/* [Parameter source code] */

} // interface1
using interface1::Parameter;
using interface1::KernelSize;
using interface1::Stride;
using interface1::Padding;
using interface1::Index;

} // namespace pooling1d
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
