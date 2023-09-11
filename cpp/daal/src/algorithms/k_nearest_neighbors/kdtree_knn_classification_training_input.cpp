/* file: kdtree_knn_classification_training_input.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of K-Nearest Neighbors (kNN) algorithm classes.
//--
*/

#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_training_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
using namespace daal::data_management;
using namespace daal::services;

services::Status Input::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter);
}

services::Status Input::checkImpl(const daal::algorithms::Parameter * parameter) const
{
    services::Status s; // Error status
    bool flag = false;  // Flag indicates error in table of labels

    data_management::NumericTablePtr dataTable = get(classifier::training::data);
    DAAL_CHECK_STATUS(s, data_management::checkNumericTable(dataTable.get(), dataStr()));

    const size_t nRows = dataTable->getNumberOfRows();

    data_management::NumericTablePtr weightsTable = get(classifier::training::weights);
    if (weightsTable)
    {
        DAAL_CHECK_STATUS(s, data_management::checkNumericTable(weightsTable.get(), weightsStr(), 0, 0, 1, nRows));
    }

    const auto par = static_cast<const kdtree_knn_classification::Parameter *>(parameter);

    if (par != NULL)
    {
        DAAL_CHECK_EX((par->nClasses > 1) && (par->nClasses < INT_MAX), services::ErrorIncorrectParameter, services::ParameterName, nClassesStr());

        if (par->resultsToEvaluate != 0)
        {
            data_management::NumericTablePtr labelsTable = get(classifier::training::labels);
            DAAL_CHECK_STATUS(s, data_management::checkNumericTable(labelsTable.get(), labelsStr(), 0, 0, 1, nRows));

            const auto nClasses = static_cast<int>(par->nClasses);

            data_management::BlockDescriptor<int> yBD;
            const_cast<data_management::NumericTable *>(labelsTable.get())->getBlockOfRows(0, nRows, data_management::readOnly, yBD);
            const int * const dy = yBD.getBlockPtr();
            for (size_t i = 0; i < nRows; ++i)
            {
                flag |= (dy[i] >= nClasses);
            }
            if (flag)
            {
                DAAL_CHECK_STATUS(s, services::Status(services::ErrorIncorrectClassLabels));
            }
            const_cast<data_management::NumericTable *>(labelsTable.get())->releaseBlockOfRows(yBD);
        }
    }

    return s;
}

} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal
