/* file: spatial_maximum_pooling2d_layer.cpp */
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
//  Implementation of spatial average pooling2d calculation algorithm and types methods.
//--
*/

#include "spatial_maximum_pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_maximum_pooling2d
{
namespace interface1
{

Parameter::Parameter(size_t pyramidHeight, size_t firstIndex, size_t secondIndex) : spatial_pooling2d::Parameter(pyramidHeight, firstIndex, secondIndex) {}

}// namespace interface1
}// namespace spatial_maximum_pooling2d
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
