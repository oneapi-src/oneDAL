/* file: pooling3d_layer.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of pooling3d calculation algorithm and types methods.
//--
*/

#include "pooling3d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace pooling3d
{
namespace interface1
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
Parameter::Parameter(size_t firstIndex, size_t secondIndex, size_t thirdIndex,
                     size_t firstKernelSize, size_t secondKernelSize, size_t thirdKernelSize,
                     size_t firstStride, size_t secondStride, size_t thirdStride,
                     size_t firstPadding, size_t secondPadding, size_t thirdPadding) :
    indices(firstIndex, secondIndex, thirdIndex), kernelSizes(firstKernelSize, secondKernelSize, thirdKernelSize),
    strides(firstStride, secondStride, thirdStride), paddings(firstPadding, secondPadding, thirdPadding)
{}

}// namespace interface1
}// namespace pooling3d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
