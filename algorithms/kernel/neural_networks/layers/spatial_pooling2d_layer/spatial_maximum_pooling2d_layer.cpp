/* file: spatial_maximum_pooling2d_layer.cpp */
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
