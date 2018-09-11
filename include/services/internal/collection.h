/* file: collection.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

#ifndef __SERVICES_INTERNAL_COLLECTION_H__
#define __SERVICES_INTERNAL_COLLECTION_H__

#include "services/base.h"
#include "services/collection.h"
#include "services/internal/error_handling_helpers.h"

namespace daal
{
namespace services
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-SERVICES__OBJECTPTRCOLLECTION"></a>
 * \brief  Class that implements functionality of collection container and holds pointers
 *         to objects of specified type, also provides automatic objects disposal
 * \tparam T Type of objects which are stored in the container
 * \tparam Deleter Type of deleter to be called on collection disposal
 */
template<typename T, typename Deleter = ObjectDeleter<T> >
class ObjectPtrCollection : public Base
{
public:
    ObjectPtrCollection() { }

    ObjectPtrCollection(const Deleter &deleter) :
        _deleter(deleter) { }

    virtual ~ObjectPtrCollection()
    {
        for (size_t i = 0; i < _objects.size(); i++)
        { _deleter( (const void *)_objects[i] ); }
    }

    T &operator [] (size_t index) const
    {
        DAAL_ASSERT( index < _objects.size() );
        return *(_objects[index]);
    }

    size_t size() const
    {
        return _objects.size();
    }

    bool push_back(T *object)
    {
        if (!object)
        { return false; }

        return _objects.safe_push_back(object);
    }

    template<typename U>
    bool safe_push_back()
    {
        return _objects.push_back(new U());
    }

private:
    ObjectPtrCollection(const ObjectPtrCollection &);
    ObjectPtrCollection &operator = (const ObjectPtrCollection &);

private:
    Deleter _deleter;
    services::Collection<T *> _objects;
};

/**
 *  <a name="DAAL-CLASS-SERVICES__INTERNAL__HEAPALLOCATABLECOLLECTION"></a>
 *  \brief   Wrapper for services::Collection that allocates and deallocates
 *           memory using internal new/delete operators
 *  \tparam  T  Type of an object stored in the container
 */
template<typename T>
class HeapAllocatableCollection : public Base, public services::Collection<T>
{
public:
    static SharedPtr<HeapAllocatableCollection<T> > create(services::Status *status = NULL)
    {
        typedef SharedPtr<HeapAllocatableCollection<T> > PtrType;

        HeapAllocatableCollection<T> *collection = new internal::HeapAllocatableCollection<T>();
        if (!collection)
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorMemoryAllocationFailed);
            return PtrType();
        }

        return PtrType(collection);
    }

    static SharedPtr<HeapAllocatableCollection<T> > create(size_t n, services::Status *status = NULL)
    {
        typedef SharedPtr<HeapAllocatableCollection<T> > PtrType;

        HeapAllocatableCollection<T> *collection = new internal::HeapAllocatableCollection<T>(n);
        if (!collection || !collection->data())
        {
            delete collection;
            services::internal::tryAssignStatusAndThrow(status, services::ErrorMemoryAllocationFailed);
            return PtrType();
        }

        return PtrType(collection);
    }

    HeapAllocatableCollection() { }

    explicit HeapAllocatableCollection(size_t n) :
        services::Collection<T>(n) { }
};

/**
 *  <a name="DAAL-CLASS-SERVICES__INTERNAL__COLLECTIONPTR"></a>
 *  \brief   Shared pointer to the Collection object
 *  \tparam  T  Type of an object stored in the container
 */
template<class T>
class CollectionPtr : public SharedPtr<HeapAllocatableCollection<T> >
{
private:
    typedef SharedPtr<HeapAllocatableCollection<T> > super;

public:
    CollectionPtr() { }

    template<class U>
    CollectionPtr(const SharedPtr<U> &other) : super(other) { }

    template<class U>
    explicit CollectionPtr(U *ptr) : super(ptr) { }

    template<class U, class D>
    explicit CollectionPtr(U *ptr, const D& deleter) : super(ptr, deleter) { }
};

} // namespace internal
} // namespace services
} // namespace daal

#endif
