/* file: dropout_layer.cpp */
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
//  Implementation of dropout calculation algorithm and types methods.
//--
*/

#include "dropout_layer_types.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace dropout
{
namespace interface1
{

Parameter::Parameter(const double retainRatio_, const size_t seed_) : retainRatio(retainRatio_), seed(seed_), engine(engines::mt19937::Batch<>::create()) {};

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    if(retainRatio <= 0.0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ArgumentName, retainRatioStr()));
    }
    return services::Status();
}

}// namespace interface1
}// namespace dropout
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
