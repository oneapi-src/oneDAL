/* file: covariance_online_parameter.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "covariance_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{

/** Default constructor */
OnlineParameter::OnlineParameter() : Parameter()
{}

/**
*  Constructs parameters of the Covariance Online algorithm by copying another parameters of the Covariance Online algorithm
*  \param[in] other    Parameters of the Covariance Online algorithm
*/
OnlineParameter::OnlineParameter(const OnlineParameter &other) : Parameter(other)
{}

/**
 * Check the correctness of the %OnlineParameter object
 */
services::Status OnlineParameter::check() const
{
    return services::Status();
}

}//namespace interface1

}//namespace covariance
}// namespace algorithms
}// namespace daal
