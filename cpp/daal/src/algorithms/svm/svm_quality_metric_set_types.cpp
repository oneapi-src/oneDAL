/* file: svm_quality_metric_set_types.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Interface for the svm algorithm quality metrics
//--
*/

#include "algorithms/svm/svm_quality_metric_set_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace quality_metric_set
{
namespace interface1
{
/**
 * Returns the result of the quality metrics algorithm
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
classifier::quality_metric::binary_confusion_matrix::ResultPtr ResultCollection::getResult(QualityMetricId id) const
{
    DAAL_ASSERT(id >= 0)
    return staticPointerCast<classifier::quality_metric::binary_confusion_matrix::Result, SerializationIface>((*this)[(size_t)id]);
}

/**
 * Returns the input object of the quality metrics algorithm
 * \param[in] id    Identifier of the input object
 * \return          %Input object that corresponds to the given identifier
 */
classifier::quality_metric::binary_confusion_matrix::InputPtr InputDataCollection::getInput(QualityMetricId id) const
{
    DAAL_ASSERT(id >= 0)
    return staticPointerCast<classifier::quality_metric::binary_confusion_matrix::Input, algorithms::Input>(
        algorithms::quality_metric_set::InputDataCollection::getInput((size_t)id));
}

} //namespace interface1
} //namespace quality_metric_set
} //namespace svm
} //namespace algorithms
} //namespace daal
