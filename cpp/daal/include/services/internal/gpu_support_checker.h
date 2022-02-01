/* file: gpu_support_checker.h */
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
//  Interface for GPU support check
//--
*/

#ifndef __GPU_SUPPORT_CHECKER_H__
#define __GPU_SUPPORT_CHECKER_H__

#include "algorithms/algorithm_container_base.h"

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

DAAL_EXPORT bool isImplementedForDevice(const services::internal::sycl::InfoDevice & deviceInfo, algorithms::AlgorithmContainerIface *);

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__TYPEREGISTRATIONCHECKERIFACE"></a>
 *  \brief Interface for algorithm container registration
 */
class TypeRegistrationCheckerIface
{
public:
    virtual bool operator()(algorithms::AlgorithmContainerIface *) = 0;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__DYNAMICTYPEREGISTRATIONCHECKER"></a>
 *  \brief Checker of algorithm container registration in runtime
 */
template <class T>
class DynamicTypeRegistrationChecker : public TypeRegistrationCheckerIface
{
public:
    DynamicTypeRegistrationChecker() {}

    virtual bool operator()(algorithms::AlgorithmContainerIface * ptr_to_check) DAAL_C11_OVERRIDE { return dynamic_cast<T *>(ptr_to_check) != NULL; }
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__GPUSUPPORTCHECKER"></a>
 *  \brief Checker whether the algorithm has GPU support
 */
class GpuSupportChecker
{
public:
    template <class T>
    void registerClass()
    {
        DynamicTypeRegistrationChecker<T> * detector_ptr = new DynamicTypeRegistrationChecker<T>();
        add(detector_ptr);
    }
    bool check(daal::algorithms::AlgorithmContainerIface * ptr_to_check)
    {
        for (Entry * it = _list.head(); it != NULL; it = it->next)
            if ((*it->checker_ptr)(ptr_to_check)) return true;
        return false;
    }
    static GpuSupportChecker & GetInstance();

private:
    GpuSupportChecker() {}
    GpuSupportChecker(const GpuSupportChecker &);
    GpuSupportChecker & operator=(const GpuSupportChecker &);

    struct Entry : public daal::Base
    {
        Entry(TypeRegistrationCheckerIface * new_checker, Entry * cur_head) : checker_ptr(new_checker), next(cur_head) {}

        TypeRegistrationCheckerIface * checker_ptr;
        Entry * next;
    };

    class List
    {
    public:
        List() : _head(NULL) {}
        ~List()
        {
            Entry * it = _head;
            while (it != NULL)
            {
                Entry * next = it->next;
                delete it;
                it = next;
            }
            _head = NULL;
        }

        void add(TypeRegistrationCheckerIface * checker_ptr)
        {
            Entry * entry = new Entry(checker_ptr, _head);
            DAAL_ASSERT(entry != NULL);
            if (entry) _head = entry;
        }
        Entry * head() { return _head; }

    private:
        Entry * _head;
        List(const List &);
        List & operator=(const List &);
    };

    void add(TypeRegistrationCheckerIface * new_checker) { _list.add(new_checker); }
    List _list;
};

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__GPUSUPPORTREGISTRAR"></a>
 *  \brief Registers Algorithm as the one has GPU support
 */
template <class T>
class GpuSupportRegistrar
{
public:
    GpuSupportRegistrar() { GpuSupportChecker::GetInstance().registerClass<T>(); }
};

/** @} */
} //namespace internal
} //namespace services
} //namespace daal

#endif // __GPU_SUPPORT_CHECKER_H__
