/* file: logistic_cross_layer.cpp */
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
//  Implementation of logistic cross calculation algorithm and types methods.
//--
*/

#include "logistic_cross_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace loss
{
namespace logistic_cross
{
namespace interface1
{
/**
*  Constructs parameters of the logistic cross-entropy layer
*/
Parameter::Parameter()
{};

/**
 * Checks the correctness of the parameter
 */
services::Status Parameter::check() const
{
    return services::Status();
}

}// namespace interface1
}// namespace logistic_cross
}// namespace loss
}// namespace layers
}// namespace neural_networks
}// namespace algorithms
}// namespace daal
