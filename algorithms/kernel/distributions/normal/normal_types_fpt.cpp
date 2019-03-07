/* file: normal_types_fpt.cpp */
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
//  Implementation of normal distribution algorithm and types methods.
//--


#include "distributions/normal/normal_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace distributions
{
namespace normal
{
namespace interface1
{
 /**
  * Check the correctness of the %Parameter object
  */
template<typename algorithmFPType>
DAAL_EXPORT services::Status Parameter<algorithmFPType>::check() const
{
    DAAL_CHECK_EX(sigma > 0, services::ErrorIncorrectParameter, services::ParameterName, sigmaStr());
    return services::Status();
}

 template DAAL_EXPORT services::Status Parameter<DAAL_FPTYPE>::check() const;

} // interface1
} // normal
} // distributions
} // algorithms
} // daal
