/* file: initializer_types_defs.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Forward declarations of classes in other namespaces.
//  Needed for implementation of neural_networks network layer initializers.
//--
*/

#ifndef __INITIALIZERS__TYPES__DEFS__H__
#define __INITIALIZERS__TYPES__DEFS__H__

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace forward
{
namespace interface1
{
class LayerIface;
typedef services::SharedPtr<LayerIface> LayerIfacePtr;
} // namespace interface1
using interface1::LayerIface;
using interface1::LayerIfacePtr;
} // namespace forward
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
