/* file: pooling1d_layer.cpp */
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
//  Implementation of pooling1d calculation algorithm and types methods.
//--
*/

#include "pooling1d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace pooling1d
{
namespace interface1
{
/**
 * Constructs the parameters of 1D pooling layer
 * \param[in] index        Index of the dimension on which pooling is performed
 * \param[in] kernelSize   Size of 1D subtensor for which the element is computed
 * \param[in] stride       Interval over the dimension on which the pooling is performed
 * \param[in] padding      Number of data elements to implicitly add to the the dimension
 *                         of the 1D subtensor on which the pooling is performed
 */
Parameter::Parameter(size_t index, size_t kernelSize, size_t stride, size_t padding) :
    index(index), kernelSize(kernelSize), stride(stride), padding(padding)
{}

}// namespace interface1
}// namespace pooling1d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
