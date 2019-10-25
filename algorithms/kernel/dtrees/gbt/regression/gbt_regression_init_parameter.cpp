/* file: gbt_regression_init_parameter.cpp */
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
//  Implementation of gbt regression classes.
//--
*/

#include "gbt_regression_init_types.h"
#include "daal_defines.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace init
{
namespace interface1
{

Parameter::Parameter(size_t _maxBins, size_t _minBinSize): maxBins(_maxBins), minBinSize(_minBinSize) {}

Parameter::Parameter(const Parameter &other): maxBins(other.maxBins), minBinSize(other.minBinSize) {}

services::Status Parameter::check() const
{
    return services::Status();
}

} // namespace interface1
} // namespace init
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
