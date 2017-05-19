/* file: average_pooling2d_layer.cpp */
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
//  Implementation of average pooling2d calculation algorithm and types methods.
//--
*/

#include "average_pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace average_pooling2d
{
namespace interface1
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
Parameter::Parameter(size_t firstIndex, size_t secondIndex, size_t firstKernelSize, size_t secondKernelSize,
          size_t firstStride, size_t secondStride, size_t firstPadding, size_t secondPadding) :
    layers::pooling2d::Parameter(firstIndex, secondIndex, firstKernelSize, secondKernelSize,
                                 firstStride, secondStride, firstPadding, secondPadding)
{}

}// namespace interface1
}// namespace average_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
