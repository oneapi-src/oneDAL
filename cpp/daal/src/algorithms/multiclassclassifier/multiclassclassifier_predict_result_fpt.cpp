/* file: multiclassclassifier_predict_result_fpt.cpp */
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
//  Implementation of classifier prediction Result.
//--
*/

#include "algorithms/multi_class_classifier/multi_class_classifier_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace prediction
{
using namespace daal::data_management;
/**
 * Allocates memory for storing prediction results of the multiclass
 * algorithm
 * \tparam  algorithmFPType     Data type for storing prediction results
 * \param[in] input     Pointer to the input objects of the classification
 * algorithm
 * \param[in] parameter Pointer to the parameters of the classification
 * algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int pmethod,
                                              const int tmethod)
{
    services::Status st;
    const Parameter * par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(par, services::ErrorNullParameterNotSupported);

    const size_t nRows    = (static_cast<const classifier::prediction::InputIface *>(input))->getNumberOfRows();
    const size_t nClasses = par->nClasses;

    if (par->resultsToEvaluate & computeClassLabels)
        set(prediction, HomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTableIface::doAllocate, 0, &st));
    if (par->resultsToEvaluate & computeDecisionFunction)
        set(decisionFunction,
            HomogenNumericTable<algorithmFPType>::create(nClasses * (nClasses - 1) / 2, nRows, NumericTableIface::doAllocate, 0, &st));
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int pmethod, const int tmethod);

} // namespace prediction
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal
