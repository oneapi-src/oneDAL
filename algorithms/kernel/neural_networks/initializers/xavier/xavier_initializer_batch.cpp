/* file: xavier_initializer_batch.cpp */
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

//++
//  Implementation of Xavier calculation functions.
//--


#include "neural_networks/initializers/xavier/xavier_initializer.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace xavier
{
namespace interface1
{

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(layer, services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, layerStr());
    return services::Status();
}

} // interface1
} // xavier
} // initializers
} // neural_networks
} // algorithms
} // daal
