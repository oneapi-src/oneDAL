/* file: profiler.h */
/*******************************************************************************
* Copyright 2019 Intel Corporation
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

namespace daal
{
namespace services
{
namespace internal
{
/**
 * @defgroup services_internal ServicesInternal
 * \brief Contains internal classes definitions
 * @{
 */

class ProfilerTask;

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__PROFILERDEFAULT"></a>
 *  \brief Profiler for measure of kernals
 */
class ProfilerDefault
{
public:
    static ProfilerTask startTask(const char * taskName);
    static void endTask(const char * taskName);
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__PROFILERTASK"></a>
 *  \brief Profiler task for measure of kernals
 */
class ProfilerTask
{
public:
    ProfilerTask(const char * taskName);
    ~ProfilerTask();

private:
    const char * _taskName;
};

} // namespace internal
} // namespace services
} // namespace daal
