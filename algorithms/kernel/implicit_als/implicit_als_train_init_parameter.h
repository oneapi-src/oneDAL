/* file: implicit_als_train_init_parameter.h */
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
//  Implementation of auxiliary implicit als methods.
//--
*/

#include "algorithms/implicit_als/implicit_als_training_init_types.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{
services::SharedPtr<data_management::HomogenNumericTable<int> > getPartition(const init::DistributedParameter * parameter, services::Status & st);

} // namespace internal
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
