/* file: status_to_error_id.h */
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

#ifndef __SERVICES_INTERNAL_STATUS_TO_ERROR_ID_H__
#define __SERVICES_INTERNAL_STATUS_TO_ERROR_ID_H__

#include "services/daal_defines.h"

namespace daal
{
namespace services
{
namespace interface1
{
class Status;
} // namespace interface1

namespace internal
{
daal::services::ErrorID DAAL_EXPORT get_error_id(const daal::services::interface1::Status & s);
} // namespace internal

} // namespace services
} // namespace daal

#endif // __SERVICES_INTERNAL_STATUS_TO_ERROR_ID_H__
