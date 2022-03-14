/** file gpu_support_checker.cpp */
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

#include "services/internal/gpu_support_checker.h"
#include "services/env_detect.h"

namespace daal
{
namespace services
{
namespace internal
{
GpuSupportChecker & GpuSupportChecker::GetInstance()
{
    static GpuSupportChecker instance;
    return instance;
}

DAAL_EXPORT bool isImplementedForDevice(const services::internal::sycl::InfoDevice & deviceInfo, algorithms::AlgorithmContainerIface * iface)
{
    bool ret = true;
    if (!deviceInfo.isCpu)
    {
        ret = GpuSupportChecker::GetInstance().check(iface);
    }
    return ret;
}

} //namespace internal
} //namespace services
} //namespace daal
