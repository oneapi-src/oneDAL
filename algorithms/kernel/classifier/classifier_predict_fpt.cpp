/* file: classifier_predict_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "classifier_predict_types.h"
#include "data_management/data/numeric_table_sycl_homogen.h"

namespace daal
{
namespace algorithms
{
namespace classifier
{
namespace prediction
{
using namespace daal::data_management;

namespace interface2
{
/**
 * Allocates memory for storing prediction results of the classification algorithm
 * \tparam  algorithmFPType     Data type for storing prediction results
 * \param[in] input     Pointer to the input objects of the classification algorithm
 * \param[in] parameter Pointer to the parameters of the classification algorithm
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    services::Status st;
    const Parameter * par = static_cast<const Parameter *>(parameter);
    DAAL_CHECK(par, services::ErrorNullParameterNotSupported);

    auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
    auto & deviceInfo = context.getInfoDevice();

    const size_t nRows    = (static_cast<const InputIface *>(input))->getNumberOfRows();
    const size_t nClasses = par->nClasses;

    if (par->resultsToEvaluate & computeClassLabels)
    {
        data_management::NumericTablePtr nt;
        if (deviceInfo.isCpu)
            nt = HomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTableIface::doAllocate, 0, &st);
        else
            nt = SyclHomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTableIface::doAllocate, 0, &st);

        set(prediction, nt);
    }
    if (par->resultsToEvaluate & computeClassProbabilities)
    {
        data_management::NumericTablePtr nt;
        if (deviceInfo.isCpu)
            nt = HomogenNumericTable<algorithmFPType>::create(nClasses, nRows, NumericTableIface::doAllocate, 0, &st);
        else
            nt = SyclHomogenNumericTable<algorithmFPType>::create(nClasses, nRows, NumericTableIface::doAllocate, 0, &st);

        set(probabilities, nt);
    }
    if (par->resultsToEvaluate & computeClassLogProbabilities)
    {
        data_management::NumericTablePtr nt;
        if (deviceInfo.isCpu)
            nt = HomogenNumericTable<algorithmFPType>::create(nClasses, nRows, NumericTableIface::doAllocate, 0, &st);
        else
            nt = SyclHomogenNumericTable<algorithmFPType>::create(nClasses, nRows, NumericTableIface::doAllocate, 0, &st);

        set(logProbabilities, nt);
    }
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace interface2

} // namespace prediction
} // namespace classifier
} // namespace algorithms
} // namespace daal
