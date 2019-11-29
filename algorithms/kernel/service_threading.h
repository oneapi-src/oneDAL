/* file: service_threading.h */
/*******************************************************************************
* Copyright 2015-2019 Intel Corporation
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
#include "threading.h"
#include "service_memory.h"
#include "service_allocators.h"

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
    void * _impl;
};

class AutoLock
{
public:
    AutoLock(Mutex & m) : _m(m) { _m.lock(); }
    ~AutoLock() { _m.unlock(); }

private:
    Mutex & _m;
};

#define AUTOLOCK(m) AutoLock __autolock(m);

template <typename F>
class task_impl : public task
{
public:
    DAAL_NEW_DELETE();
    virtual void run() { _func(); }
    virtual void destroy() { delete this; }
    static task_impl<F> * create(const F & o) { return new task_impl<F>(o); }

private:
    task_impl(const F & o) : task(), _func(o) {}
    F _func;
};

class task_group
{
public:
    task_group() : _impl(nullptr) { _impl = _daal_new_task_group(); }
    ~task_group()
    {
        if (_impl) _daal_del_task_group(_impl);
    }
    template <typename F>
    void run(F & f)
    {
        if (_impl)
            _daal_run_task_group(_impl, task_impl<F>::create(f));
        else
            f();
    }
    void wait()
    {
        if (_impl) _daal_wait_task_group(_impl);
    }

protected:
    void * _impl;
};

template <typename T, CpuType cpu, typename Allocator = services::internal::ScalableMalloc<T, cpu> >
class TlsMem : public daal::tls<T *>
{
public:
    typedef daal::tls<T *> super;
    TlsMem(size_t n) : super([=]() -> T * { return Allocator::allocate(n); }) {}
    ~TlsMem()
    {
        this->reduce([](T * ptr) -> void {
            if (ptr) Allocator::deallocate(ptr);
        });
    }
};

template <typename algorithmFPType, CpuType cpu>
class TlsSum : public daal::TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> >
{
public:
    typedef daal::TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> > super;
    TlsSum(size_t n) : super(n) {}
    void reduceTo(algorithmFPType * res, size_t n)
    {
        bool bFirst = true;
        this->reduce([=, &bFirst](algorithmFPType * ptr) -> void {
            if (!ptr) return;
            if (bFirst)
            {
                for (size_t i = 0; i < n; ++i) res[i] = ptr[i];
                bFirst = false;
            }
            else
            {
                for (size_t i = 0; i < n; ++i) res[i] += ptr[i];
            }
        });
    }
};

} // namespace daal

#endif
