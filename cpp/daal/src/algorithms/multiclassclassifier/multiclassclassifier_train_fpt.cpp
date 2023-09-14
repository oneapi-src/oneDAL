/* file: multiclassclassifier_train_fpt.cpp */
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
//  Implementation of multi class classifier algorithm and types methods.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_train_types.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace interface1
{
/**
 * Registers user-allocated memory to store the results of the multi-class classifier training decomposition
 * \param[in] input       Pointer to the structure with input objects
 * \param[in] parameter   Pointer to the structure with algorithm parameters
 * \param[in] method      Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const classifier::training::Input * algInput = static_cast<const classifier::training::Input *>(input);
    services::Status st;
    const multi_class_classifier::interface1::ParameterBase * algParameter1 =
        dynamic_cast<const multi_class_classifier::interface1::ParameterBase *>(parameter);
    if (algParameter1)
    {
        multi_class_classifier::ModelPtr modelPtr = Model::create(algInput->getNumberOfFeatures(), algParameter1, &st);
        set(classifier::training::model, modelPtr);
    }
    else
    {
        const ParameterBase * algParameter2 = dynamic_cast<const ParameterBase *>(parameter);
        DAAL_CHECK(algParameter2, services::ErrorNullParameterNotSupported);
        ModelPtr modelPtr = Model::create(algInput->getNumberOfFeatures(), algParameter2, &st);
        set(classifier::training::model, modelPtr);
    }
    DAAL_CHECK_STATUS_VAR(st);
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input,
                                                                    const daal::algorithms::Parameter * parameter, const int method);

} // namespace interface1
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
