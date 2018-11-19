/* file: pooling3d_layer.cpp */
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
