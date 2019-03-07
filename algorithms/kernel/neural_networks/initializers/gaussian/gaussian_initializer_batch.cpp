/* file: gaussian_initializer_batch.cpp */
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

//++
//  Implementation of gaussian calculation functions.
//--


#include "neural_networks/initializers/gaussian/gaussian_initializer.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace gaussian
{
namespace interface1
{

services::Status Parameter::check() const
{
    DAAL_CHECK_EX(sigma > 0, services::ErrorIncorrectParameter, services::ParameterName, sigmaStr());
    return services::Status();
}

} // interface1
} // gaussian
} // initializers
} // neural_networks
} // algorithms
} // daal
