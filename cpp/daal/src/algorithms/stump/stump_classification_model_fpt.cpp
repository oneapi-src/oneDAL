/* file: stump_classification_model_fpt.cpp */
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
//  Implementation of the decision stump model constructor.
//--
*/

#include "algorithms/stump/stump_classification_model.h"
#include "src/algorithms/stump/stump_classification_model_visitor.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace classification
{
template <typename modelFPType>
DAAL_EXPORT modelFPType Model::getSplitValue()
{
    StumpNodeVisitor visitor(_nClasses);
    traverseDFS(visitor);
    return (modelFPType)visitor.splitValue;
}

template <typename modelFPType>
DAAL_EXPORT modelFPType Model::getLeftValue()
{
    StumpNodeVisitor visitor(_nClasses);
    traverseDFS(visitor);
    return (modelFPType)visitor.leftValue;
}

template <typename modelFPType>
DAAL_EXPORT modelFPType Model::getRightValue()
{
    StumpNodeVisitor visitor(_nClasses);
    traverseDFS(visitor);
    return (modelFPType)visitor.rightValue;
}

template DAAL_EXPORT DAAL_FPTYPE Model::getSplitValue();
template DAAL_EXPORT DAAL_FPTYPE Model::getLeftValue();
template DAAL_EXPORT DAAL_FPTYPE Model::getRightValue();

} // namespace classification
} // namespace stump
} // namespace algorithms
} // namespace daal
