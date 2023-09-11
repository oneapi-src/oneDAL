/* file: naivebayes_model.cpp */
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
//  Implementation of multinomial naive bayes algorithm parameters.
//--
*/

#include "algorithms/naive_bayes/multinomial_naive_bayes_model.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
PartialModel::PartialModel() : _nObservations(0) {}

Status Parameter::check() const
{
    Status s;
    DAAL_CHECK_STATUS(s, classifier::Parameter::check());

    if (priorClassEstimates)
    {
        s |= checkNumericTable(priorClassEstimates.get(), priorClassEstimatesStr(), 0, 0, 1, nClasses);
    }

    return s;
}

} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
