/* file: gbt_training_parameter.h */
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
//  Implementation of gradient boosted trees algorithm classes.
//--
*/

#ifndef __GBT_TRAINING_PARAMETER_KERNEL_H__
#define __GBT_TRAINING_PARAMETER_KERNEL_H__

#include "algorithms/gradient_boosted_trees/gbt_classification_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
Status checkImpl(const gbt::training::Parameter & prm);
}
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif
