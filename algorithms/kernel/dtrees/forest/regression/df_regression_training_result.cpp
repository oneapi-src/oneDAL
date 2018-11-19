/* file: df_regression_training_result.cpp */
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
//  Implementation of decision forest algorithm classes.
//--
*/

#include "df_regression_training_types_result.h"
#include "serialization_utils.h"
#include "daal_strings.h"
#include "daal_strings.h"
using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_FOREST_REGRESSION_TRAINING_RESULT_ID);

Result::Result() : algorithms::regression::training::Result(lastResultNumericTableId + 1)
{
    _impl = new Result::ResultImpl();
}

Result::~Result() { delete _impl; }

Result::Result( const Result& other ): algorithms::regression::training::Result( other )
{
    _impl = new Result::ResultImpl(*other._impl);
}

decision_forest::regression::ModelPtr Result::get(ResultId id) const
{
    return decision_forest::regression::Model::cast(
        algorithms::regression::training::Result::get(algorithms::regression::training::ResultId(id)));
}

void Result::set(ResultId id, const decision_forest::regression::ModelPtr &value)
{
    algorithms::regression::training::Result::set(algorithms::regression::training::ResultId(id), value);
}

NumericTablePtr Result::get(ResultNumericTableId id) const
{
    return NumericTable::cast(Argument::get(id));
}

void Result::set(ResultNumericTableId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, algorithms::regression::training::Result::check(input, par, method));
    const decision_forest::regression::training::Input *algInput = static_cast<const decision_forest::regression::training::Input *>(input);

    //TODO: check model
    const Parameter* algParameter = static_cast<const Parameter *>(par);
    if(algParameter->resultsToCompute & decision_forest::training::computeOutOfBagError)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(outOfBagError).get(), outOfBagErrorStr(), 0, 0, 1, 1));
    }
    if(algParameter->resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
    {
        const auto nObs = algInput->get(decision_forest::regression::training::data)->getNumberOfRows();
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(outOfBagErrorPerObservation).get(), outOfBagErrorPerObservationStr(), 0, 0, 1, nObs));
    }
    if(algParameter->varImportance != decision_forest::training::none)
    {
        const auto nFeatures = algInput->get(decision_forest::regression::training::data)->getNumberOfColumns();
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(variableImportance).get(), variableImportanceStr(), 0, 0, nFeatures, 1));
    }
    return s;
}

engines::EnginePtr Result::get(ResultEngineId id) const
{
    return _impl->getEngine();
}

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
