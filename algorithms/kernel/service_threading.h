/* file: service_threading.h */
/*******************************************************************************
* Copyright 2015-2017 Intel Corporation
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
//  Declaration of service threding classes and utilities
//--
*/
#ifndef __SERVICE_THREADING_H__
#define __SERVICE_THREADING_H__

namespace daal
{

class Mutex
{
public:
    Mutex();
    ~Mutex();
    void lock();
    void unlock();
private:
    void* _impl;
};

class AutoLock
{
public:
    AutoLock(Mutex& m) : _m(m){ _m.lock(); }
    ~AutoLock() { _m.unlock(); }
private:
    Mutex& _m;
};

#define AUTOLOCK(m) AutoLock __autolock(m);

} // namespace daal

#endif
