/* file: naivebayes_train_partial_result.h */
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
//  Implementation of multinomial naive bayes algorithm and types methods.
//--
*/

#include "algorithms/naive_bayes/multinomial_naive_bayes_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
/**
 * Allocates memory for storing partial results of the naive Bayes training algorithm
 * \param[in] input        Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status PartialResult::initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    get(classifier::training::partialModel)->initialize<algorithmFPType>();
    return Status();
}

/**
 * Allocates memory for storing partial results of the naive Bayes training algorithm
 * \param[in] input        Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status PartialResult::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const classifier::training::InputIface * algInput = static_cast<const classifier::training::InputIface *>(input);
    size_t nFeatures                                  = algInput->getNumberOfFeatures();
    Status st;
    PartialModelPtr partialModelPtr;

    const multinomial_naive_bayes::Parameter * algPar = dynamic_cast<const multinomial_naive_bayes::Parameter *>(parameter);
    if (algPar) partialModelPtr = PartialModel::create<algorithmFPType>(nFeatures, *algPar, &st);
    DAAL_CHECK(partialModelPtr, ErrorNullPartialModel);
    DAAL_CHECK_STATUS_VAR(st);
    set(classifier::training::partialModel, partialModelPtr);
    return st;
}

} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
