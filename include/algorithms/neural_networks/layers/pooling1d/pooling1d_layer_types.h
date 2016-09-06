/* file: pooling1d_layer_types.h */
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
struct Parameter: public layers::Parameter
{
    /**
     * Constructs the parameters of 1D pooling layer
     * \param[in] index        Index of the dimension on which pooling is performed
     * \param[in] kernelSize   Size of 1D subtensor for which the element is computed
     * \param[in] stride       Interval over the dimension on which the pooling is performed
     * \param[in] padding      Number of data elements to implicitly add to the the dimension
     *                         of the 1D subtensor on which the pooling is performed
     */
    Parameter(size_t index, size_t kernelSize = 2, size_t stride = 2, size_t padding = 0) :
        index(index), kernelSize(kernelSize), stride(stride), padding(padding)
    {}

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
