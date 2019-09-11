/* file: zscore_v2.cpp */
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "zscore_types_v2.h"
#include "zscore_result_v1.h"
#include "serialization_utils.h"
#include "daal_strings.h"
#include "service_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{

namespace interface2
{

BaseParameter::BaseParameter() : resultsToCompute(none) {}

}// namespace interface2

}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal
