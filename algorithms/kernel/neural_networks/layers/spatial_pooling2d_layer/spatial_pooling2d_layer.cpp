/* file: spatial_pooling2d_layer.cpp */
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
//  Implementation of spatial pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_pooling2d
{
namespace interface1
{
/**
 * Constructs the parameters of 2D spatial layer
 * \param[in] _pyramidHeight     The value of pyramid height
 * \param[in] firstIndex         Index of the first of two dimensions on which the spatial is performed
 * \param[in] secondIndex        Index of the second of two dimensions on which the spatial is performed
 */
Parameter::Parameter(size_t _pyramidHeight, size_t firstIndex, size_t secondIndex) :
    pyramidHeight(_pyramidHeight), indices(firstIndex, secondIndex)
{}

}// namespace interface1
}// namespace spatial_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
