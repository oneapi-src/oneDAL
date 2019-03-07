/* file: split_layer.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of split calculation algorithm and types methods.
//--
*/

#include "split_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace split
{
namespace interface1
{
/**
*  Constructs parameters of the forward split layer
*  \param[in] nOutputs   Number of outputs for forward split layer
*  \param[in] nInputs    Number of inputs for backward split layer
*/
Parameter::Parameter(size_t nOutputs, size_t nInputs) : nOutputs(nOutputs), nInputs(nInputs) {};

}// namespace interface1
}// namespace split
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
