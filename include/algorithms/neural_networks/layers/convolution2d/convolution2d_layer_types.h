/* file: convolution2d_layer_types.h */
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
//  Two-dimensional (2D) convolution layer parameter structure.
//--
*/

#ifndef __CONVOLUTION2D_LAYER_TYPES_H__
#define __CONVOLUTION2D_LAYER_TYPES_H__

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
 * @defgroup convolution2d Two-dimensional __onvolution Layer
 * \copydoc daal::algorithms::neural_networks::layers::convolution2d
 * @ingroup layers
 * @{
 */
namespace convolution2d
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__METHOD"></a>
 * Available methods to compute forward and backward 2D convolution layer
 */
enum Method
{
    defaultDense = 0,    /*!< Default: performance-oriented method. */
};

/**
 * Available identifiers of results of the forward 2D convolution layer
 * and input objects for the backward 2D convolution layer
 */
enum LayerDataId
{
    auxData    = 0, /*!< Data processed at the forward stage of the layer */
    auxWeights = 1, /*!< Input weights for forward stage of the layer */
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__KERNELSIZE"></a>
 * \brief Data structure representing the size of the two-dimensional kernel subtensor
 */
struct KernelSizes
{
    /**
    * Constructs the structure representing the size of the two-dimensional kernel subtensor
    * \param[in]  first  Size of the first dimension of the two-dimensional kernel subtensor
    * \param[in]  second Size of the second dimension of the wto-dimensional kernel subtensor
    */
    KernelSizes(size_t first, size_t second) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__STRIDE"></a>
 * \brief Data structure representing the intervals on which the subtensors for 2D convolution are selected
 */
struct Strides
{
    /**
    * Constructs the structure representing the intervals on which the subtensors for 2D convolution are selected
    * \param[in]  first  Interval over the first dimension on which the 2D convolution is performed
    * \param[in]  second Interval over the second dimension on which the 2D convolution is performed
    */
    Strides(size_t first, size_t second) { size[0] = first; size[1] = second; }
    size_t size[2];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__PADDING"></a>
 * \brief Data structure representing the number of data elements to implicitly add
 *        to each size of the two-dimensional subtensor on which 2D convolution is performed
 */
struct Paddings
{
    /**
    * Constructs the structure representing the number of data elements to implicitly add
    * to each size of the two-dimensional subtensor on which 2D convolution is performed
    * \param[in]  first  Number of data elements to add to the the first dimension of the two-dimensional subtensor
    * \param[in]  second Number of data elements to add to the the second dimension of the two-dimensional subtensor
    */
    Paddings(int first, int second) { size[0] = first; size[1] = second; }
    int size[2];
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__SPATIALDIMENSIONS"></a>
 * \brief Data structure representing the indices of the two dimensions on which 2D convolution is performed
 */
struct Indices
{
    /**
    * Constructs the structure representing the indices of the two dimensions on which 2D convolution is performed
    * \param[in]  first  The first dimension index
    * \param[in]  second The second dimension index
    */
    Indices(size_t first, size_t second) { dims[0] = first; dims[1] = second; }
    size_t dims[2];
};

namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__CONVOLUTION2D__PARAMETER"></a>
 * \brief 2D convolution layer parameters
 */
class Parameter: public layers::Parameter
{
public:
    /**
     *  Default constructor
     */
    Parameter() : groupDimension(1), indices(2, 3), kernelSizes(2, 2), strides(2, 2), paddings(0, 0), nKernels(1), nGroups(1) {}

    Indices indices;         /*!< Data structure representing the dimension for convolution kernels. (2,3) is supported now */
    size_t groupDimension;   /*!< Dimension for which the grouping is applied. groupDimension=1 is supported now */
    KernelSizes kernelSizes; /*!< Data structure representing the sizes of the two-dimensional kernel subtensor */
    Strides strides;         /*!< Data structure representing the intervals on which the kernel should be applied to the input */
    Paddings paddings;       /*!< Data structure representing the number of data to be implicitly added to the subtensor */
    size_t nKernels;         /*!< Number of kernels applied to the input layer data */
    size_t nGroups;          /*!< Number of groups which the input data is split in groupDimension dimension */
};

} // namespace interface1
using interface1::Parameter;

} // namespace convolution2d
/** @} */
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
