/* file: spatial_pooling2d_layer.cpp */
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
