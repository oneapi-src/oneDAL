/** file service_algo_utils.cpp */
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
//
//--
*/

#include "src/services/service_utils.h"
#include "src/services/service_algo_utils.h"
#include "services/error_indexes.h"
#include "services/error_handling.h"

namespace daal
{
namespace services
{
namespace interface1
{
HostAppIface::HostAppIface() : _impl(nullptr) {}

HostAppIface::~HostAppIface()
{
    delete _impl;
    _impl = NULL;
}
} // namespace interface1

namespace internal
{
bool isCancelled(services::Status & s, services::HostAppIface * pHostApp)
{
    if (!pHostApp || !pHostApp->isCancelled()) return false;
    s.add(services::ErrorUserCancelled);
    return true;
}

HostAppHelper::HostAppHelper(HostAppIface * hostApp, size_t maxJobsBeforeCheck)
    : _hostApp(hostApp), _maxJobsBeforeCheck(maxJobsBeforeCheck), _nJobsAfterLastCheck(0)
{}

bool HostAppHelper::isCancelled(services::Status & s, size_t nJobsToDo)
{
    if (!_hostApp) return false;
    _nJobsAfterLastCheck += nJobsToDo;
    if (_nJobsAfterLastCheck < _maxJobsBeforeCheck) return false;
    _nJobsAfterLastCheck = 0;
    return services::internal::isCancelled(s, _hostApp);
}

void HostAppHelper::setup(size_t maxJobsBeforeCheck)
{
    _maxJobsBeforeCheck = maxJobsBeforeCheck;
}

void HostAppHelper::reset(size_t maxJobsBeforeCheck)
{
    setup(maxJobsBeforeCheck);
    _nJobsAfterLastCheck = 0;
}

} // namespace internal
} // namespace services
} // namespace daal
