/* file: df_classification_training_types.cpp */
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

#include "algorithms/decision_forest/decision_forest_classification_training_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{

namespace training { services::Status checkImpl(const decision_forest::training::Parameter& prm); }

namespace classification
{
namespace training
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_DECISION_FOREST_CLASSIFICATION_TRAINING_RESULT_ID);
Result::Result() : classifier::training::Result(lastResultId + 1){}

daal::algorithms::decision_forest::classification::ModelPtr Result::get(classifier::training::ResultId id) const
{
    return staticPointerCast<daal::algorithms::decision_forest::classification::Model, SerializationIface>(Argument::get(id));
}

void Result::set(classifier::training::ResultId id, const ModelPtr &value)
{
    Argument::set(id, value);
}

NumericTablePtr Result::get(ResultNumericTableId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Result::set(ResultNumericTableId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    DAAL_CHECK(Argument::size() == lastResultId + 1, ErrorIncorrectNumberOfOutputNumericTables);
    const classifier::training::Input *algInput = static_cast<const classifier::training::Input *>(input);

    ModelPtr m = get(classifier::training::model);
    DAAL_CHECK(m.get(), ErrorNullModel);

    services::Status s;
    const Parameter* algParameter = static_cast<const Parameter *>(par);
    if(algParameter->resultsToCompute & decision_forest::training::computeOutOfBagError)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(outOfBagError).get(), outOfBagErrorStr(), 0, 0, 1, 1));
    }
    if(algParameter->resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation)
    {
        const auto nObs = algInput->get(classifier::training::data)->getNumberOfRows();
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(outOfBagErrorPerObservation).get(), outOfBagErrorPerObservationStr(), 0, 0, 1, nObs));
    }
    if(algParameter->varImportance != decision_forest::training::none)
    {
        const auto nFeatures = algInput->get(classifier::training::data)->getNumberOfColumns();
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(get(variableImportance).get(), variableImportanceStr(), 0, 0, nFeatures, 1));
    }
    return s;
}

services::Status Parameter::check() const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());
    DAAL_CHECK_STATUS(s, decision_forest::training::checkImpl(*this));
    return s;
}

} // namespace interface1
} // namespace training
} // namespace classification
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
