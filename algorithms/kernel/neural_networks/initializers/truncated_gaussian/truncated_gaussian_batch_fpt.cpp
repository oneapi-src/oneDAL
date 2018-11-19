/* file: truncated_gaussian_batch_fpt.cpp */
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
//  Implementation of truncated gaussian initializer functions.
//--


#include "neural_networks/initializers/truncated_gaussian/truncated_gaussian_initializer.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace truncated_gaussian
{
namespace interface1
{

template<typename algorithmFPType>
DAAL_EXPORT Parameter<algorithmFPType>::Parameter(double _mean, double _sigma, size_t _seed) :
    mean(_mean), sigma(_sigma), seed(_seed)
{
    a = (algorithmFPType)(mean - 2.0 * sigma);
    b = (algorithmFPType)(mean + 2.0 * sigma);
}

template<typename algorithmFPType>
DAAL_EXPORT services::Status Parameter<algorithmFPType>::check() const
{
    DAAL_CHECK_EX(a < b, services::ErrorIncorrectParameter, services::ParameterName, aStr());
    DAAL_CHECK_EX(sigma > 0, services::ErrorIncorrectParameter, services::ParameterName, sigmaStr());
    return services::Status();
}

template DAAL_EXPORT Parameter<DAAL_FPTYPE>::Parameter(double mean, double sigma, size_t seed);
template DAAL_EXPORT services::Status Parameter<DAAL_FPTYPE>::check() const;

} // interface1
} // truncated_gaussian
} // initializers
} // neural_networks
} // algorithms
} // daal
