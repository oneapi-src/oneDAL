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

#ifndef __DAAL_SYCL_H__
#define __DAAL_SYCL_H__

#include <sycl/sycl.hpp>

#define DAAL_SYCL_INTERFACE
#include "daal.h"

#include "services/internal/execution_context.h"
#include "data_management/data/internal/numeric_table_sycl_homogen.h"
#include "data_management/data/internal/numeric_table_sycl_soa.h"
#include "data_management/data/internal/numeric_table_sycl_csr.h"

namespace daal
{
namespace services
{
using services::internal::Buffer;
using services::internal::ExecutionContext;
using services::internal::SyclExecutionContext;
using services::internal::CpuExecutionContext;

} // namespace services
} // namespace daal

namespace daal
{
namespace data_management
{
using data_management::internal::SyclNumericTable;
using data_management::internal::SyclNumericTablePtr;
using data_management::internal::SyclHomogenNumericTable;
using data_management::internal::SyclSOANumericTable;
using data_management::internal::SyclSOANumericTablePtr;
using data_management::internal::SyclCSRNumericTable;
using data_management::internal::SyclCSRNumericTablePtr;

} // namespace data_management
} // namespace daal

#endif
