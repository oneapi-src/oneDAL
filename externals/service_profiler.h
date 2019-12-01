/* file: service_profiler.h */
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
//  Profiler for measure of kernals
//--
*/

namespace daal
{
namespace internal
{
class ProfilerTask
{
public:
    ProfilerTask(const char * task_name);
    ~ProfilerTask();

private:
    const char * _task_name;
};

class Profiler
{
public:
    static ProfilerTask startTask(const char * task_name);
    static void endTask(const char * task_name);
};

} // namespace internal
} // namespace daal
