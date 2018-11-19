/* file: fullyconnected_layer.cpp */
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
//  Implementation of fullyconnected calculation algorithm and types methods.
//--
*/

#include "fullyconnected_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
namespace interface1
{
/**
 *  Main constructor
 *  \param[in] _nOutputs A number of layer outputs m. The parameter required to initialize the layer
 */
Parameter::Parameter(size_t _nOutputs) : nOutputs(_nOutputs) {}

}// namespace interface1
}// namespace fullyconnected
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
