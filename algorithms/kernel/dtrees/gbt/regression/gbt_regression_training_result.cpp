/* file: gbt_regression_training_result.cpp */
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
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_regression_training_types.h"
#include "serialization_utils.h"
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
namespace training
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_GBT_REGRESSION_TRAINING_RESULT_ID);
Result::Result() : algorithms::regression::training::Result(lastResultId + 1) {};

gbt::regression::ModelPtr Result::get(ResultId id) const
{
    return gbt::regression::Model::cast(
        algorithms::regression::training::Result::get(algorithms::regression::training::ResultId(id)));
}

void Result::set(ResultId id, const gbt::regression::ModelPtr &value)
{
    algorithms::regression::training::Result::set(algorithms::regression::training::ResultId(id), value);
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    return algorithms::regression::training::Result::check(input, par, method);
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
