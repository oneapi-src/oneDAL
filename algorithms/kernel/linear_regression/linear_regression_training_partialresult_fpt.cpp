/* file: linear_regression_training_partialresult_fpt.cpp */
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
//  Implementation of the linear regression algorithm interface
//--
*/

#include "linear_regression_training_partialresult.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{

template DAAL_EXPORT services::Status PartialResult::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status PartialResult::initialize<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace training
}// namespace linear_regression
}// namespace algorithms
}// namespace daal
