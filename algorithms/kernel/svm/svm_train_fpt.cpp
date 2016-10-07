/* file: svm_train_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of cholesky algorithm and types methods.
//--
*/

#include "svm_train_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace interface1
{
/**
 * Allocates memory for storing SVM training results
 * \param[in] input     Pointer to input structure
 * \param[in] parameter Pointer to parameter structure
 * \param[in] method    Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const classifier::training::Input *algInput = static_cast<const classifier::training::Input *>(input);

    algorithmFPType dummy = 1.0;
    set(classifier::training::model, services::SharedPtr<svm::Model>(
            new svm::Model(dummy, algInput->get(classifier::training::data)->getDataLayout())));
}

template DAAL_EXPORT void Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace svm
}// namespace cholesky
}// namespace algorithms
}// namespace daal
