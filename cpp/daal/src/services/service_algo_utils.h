/* file: service_algo_utils.h */
/*******************************************************************************
* Copyright 2015 Intel Corporation
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
//  Declaration of service utilities used with services structures
//--
*/
#ifndef __SERVICE_ALGO_UTILS_H__
#define __SERVICE_ALGO_UTILS_H__

#include "services/host_app.h"
#include "services/error_handling.h"

namespace daal
{
namespace algorithms
{
class AlgorithmIface;
class Input;
} // namespace algorithms

namespace services
{
namespace internal
{
DAAL_EXPORT services::HostAppIface * hostApp(algorithms::Input & inp);
void setHostApp(const services::SharedPtr<services::HostAppIface> & pHostApp, algorithms::Input & inp);
services::HostAppIfacePtr getHostApp(daal::algorithms::Input & inp);
bool isCancelled(services::Status & s, services::HostAppIface * pHostApp);

//////////////////////////////////////////////////////////////////////////////////////////
// Helper class handling cancellation status depending on the number of jobs to be done
//////////////////////////////////////////////////////////////////////////////////////////
class HostAppHelper
{
public:
    HostAppHelper(HostAppIface * hostApp, size_t maxJobsBeforeCheck);
    bool isCancelled(services::Status & s, size_t nJobsToDo);
    void setup(size_t maxJobsBeforeCheck);
    void reset(size_t maxJobsBeforeCheck);

private:
    services::HostAppIface * _hostApp;
    size_t _maxJobsBeforeCheck; //granularity
    size_t _nJobsAfterLastCheck;
};

} // namespace internal
} // namespace services
} // namespace daal

#endif
