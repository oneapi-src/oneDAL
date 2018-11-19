/* file: initializer_types_defs.h */
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
