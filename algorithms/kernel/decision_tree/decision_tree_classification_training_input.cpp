/* file: decision_tree_classification_training_input.cpp */
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
//  Implementation of Decision tree algorithm classes.
//--
*/

#include "algorithms/decision_tree/decision_tree_classification_training_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace decision_tree
{
namespace classification
{
namespace training
{
namespace interface1
{

using namespace daal::data_management;
using namespace daal::services;

Input::Input() : classifier::training::Input(lastInputId + 1) {}

NumericTablePtr Input::get(decision_tree::classification::training::InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

void Input::set(decision_tree::classification::training::InputId id, const data_management::NumericTablePtr & value)
{
    Argument::set(id, value);
}

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    services::Status s;
    DAAL_CHECK_STATUS(s, classifier::training::Input::check(parameter, method));
    return checkImpl(parameter);
}

services::Status Input::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    services::Status s;
    const decision_tree::classification::Parameter * const par = static_cast<const decision_tree::classification::Parameter *>(parameter);

    if (par->pruning == decision_tree::reducedErrorPruning)
    {
        const NumericTablePtr dataForPruningTable = get(dataForPruning);
        DAAL_CHECK_STATUS(s, checkNumericTable(dataForPruningTable.get(), dataForPruningStr(), 0, 0, this->getNumberOfFeatures()));
        const int unexpectedLabelsLayouts = (int)NumericTableIface::upperPackedSymmetricMatrix
                                            | (int)NumericTableIface::lowerPackedSymmetricMatrix
                                            | (int)NumericTableIface::upperPackedTriangularMatrix
                                            | (int)NumericTableIface::lowerPackedTriangularMatrix;
        DAAL_CHECK_STATUS(s, checkNumericTable(get(labelsForPruning).get(), labelsForPruningStr(), unexpectedLabelsLayouts, 0, 1,
                                               dataForPruningTable->getNumberOfRows()));
    }
    else
    {
        DAAL_CHECK_EX(get(dataForPruning).get() == nullptr, ErrorIncorrectOptionalInput, ArgumentName, dataForPruningStr());
        DAAL_CHECK_EX(get(labelsForPruning).get() == nullptr, ErrorIncorrectOptionalInput, ArgumentName, labelsForPruningStr());
    }

    return s;
}

} // namespace interface1

using interface1::Input;

} // namespace training
} // namespace classification
} // namespace decision_tree
} // namespace algorithms
} // namespace daal
