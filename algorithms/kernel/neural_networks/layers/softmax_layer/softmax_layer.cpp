/* file: softmax_layer.cpp */
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
