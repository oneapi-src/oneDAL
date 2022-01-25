/* file: execution_context.cpp */
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

#include "services/internal/execution_context.h"
#include "services/env_detect.h"

namespace daal
{
namespace services
{
namespace internal
{
namespace interface1
{
sycl::ExecutionContextIface & getDefaultContext()
{
    return services::Environment::getInstance()->getDefaultExecutionContext();
}

} // namespace interface1
} // namespace internal
} // namespace services
} // namespace daal
