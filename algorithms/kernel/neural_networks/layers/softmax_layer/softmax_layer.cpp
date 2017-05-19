/* file: softmax_layer.cpp */
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
//  Implementation of softmax calculation algorithm and types methods.
//--
*/

#include "softmax_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace softmax
{
namespace interface1
{
/**
 *  Constructs parameters of the softmax layer
 *  \param[in] _dimension   Dimension index to calculate softmax
 */
Parameter::Parameter(size_t _dimension) : dimension(_dimension) {}

/**
 *  Constructs parameters of the softmax layer by copying another parameters of the softmax layer
 *  \param[in] other    Parameters of the softmax layer
 */
Parameter::Parameter(const Parameter &other) : dimension(other.dimension) {}

}// namespace interface1
}// namespace softmax
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
