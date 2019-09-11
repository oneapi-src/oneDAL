/* file: service_unique_ptr.h */
/*******************************************************************************
* Copyright 2015-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#ifndef __SERVICE_UNIQUE_PTR_H__
#define __SERVICE_UNIQUE_PTR_H__

#include "service_utils.h"
#include "service_allocators.h"

namespace daal
{
namespace internal
{

/* STL compatible unique_ptr implementation (doesn't handle the case when T is an array) */
template<typename T, CpuType cpu,
         typename Deleter = services::internal::DefaultDeleter<T, cpu>>
class UniquePtr
{
private:
    T *_object;
    Deleter _deleter;

public:
    UniquePtr() : _object(nullptr) { }

    explicit UniquePtr(T *object) : _object(object) { }

    template<typename U, typename UDeleter>
    UniquePtr(UniquePtr<U, cpu, UDeleter> &&other) :
        _object( other.release() ),
        _deleter( services::internal::forward<cpu, UDeleter>(other.getDeleter()) ) { }

    ~UniquePtr() { reset(); }

    inline T *get() const { return _object; }
    inline bool operator () () const { return _object != nullptr; }

    inline T &operator *  () const { return *_object; }
    inline T *operator -> () const { return _object;  }

    template<typename U, typename UDeleter>
    UniquePtr &operator = (UniquePtr<U, cpu, UDeleter> &&other)
    {
        reset(other.release());
        _deleter = services::internal::move<cpu, UDeleter>(other.getDeleter());
        return *this;
    }

    inline void reset(T *object = nullptr)
    {
        if (_object) { _deleter(_object); }
        _object = object;
    }

    inline T *release()
    {
        T *result = _object;
        _object = nullptr;
        return result;
    }

    inline       Deleter &getDeleter()       { return _deleter; }
    inline const Deleter &getDeleter() const { return _deleter; }

    template<typename U, typename UDeleter>
    UniquePtr(const UniquePtr<U, cpu, UDeleter> &) = delete;

    template<typename U, typename UDeleter>
    UniquePtr &operator = (const UniquePtr<U, cpu, UDeleter> &) = delete;
};

// Creates UniquePtr<T, cpu> by calling constructor of T with the given arguments
// Usage: auto object = makeUnique<T, cpu>(arg_1, ..., arg_2);
template<typename T, CpuType cpu, typename ...Args>
UniquePtr<T, cpu> makeUnique(Args &&...args)
{
    using namespace daal::services::internal;
    return UniquePtr<T, cpu>( new T(forward<Args>(args) ...) );
}

} // namespace internal
} // namespace daal

#endif
