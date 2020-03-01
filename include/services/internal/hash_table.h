/* file: hash_table.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Hash table implementation
//--
*/

#ifndef __DAAL_ONEAPI_INTERNAL_HASH_TABLE_H__
#define __DAAL_ONEAPI_INTERNAL_HASH_TABLE_H__

#include "services/daal_string.h"
#include "services/internal/error_handling_helpers.h"
#include "services/error_indexes.h"

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

/**
 *  <a name="DAAL-CLASS-SERVICES-INTERNAL__HASHTABLE"></a>
 *  \brief Hash table implementation
 */
template <class T, size_t size>
class HashTable
{
public:
    HashTable()
    {
        for (size_t i = 0; i < size; ++i) _arr[i] = nullptr;
    }

    ~HashTable()
    {
        for (size_t i = 0; i < size; ++i)
        {
            delete _arr[i];
            _arr[i] = nullptr;
        }
    }

    bool contain(const services::String & key, services::Status & status)
    {
        size_t id = get_unique_hash(key, status);
        if (!status.ok()) return false;
        return _arr[id] != nullptr;
    }

    void add(const services::String & key, const services::SharedPtr<T> obj, services::Status & status)
    {
        size_t id = get_unique_hash(key, status);
        if (!status.ok()) return;
        _arr[id] = new Entry(obj, key);
    }

    services::SharedPtr<T> get(const services::String & key, services::Status & status)
    {
        size_t id = get_unique_hash(key, status);
        return status.ok() == true ? _arr[id]->obj : services::SharedPtr<T>();
    }

private:
    struct Entry
    {
        Entry() : obj(), key() {}
        Entry(const services::SharedPtr<T> & obj_, const services::String & key_) : obj(obj_), key(key_) {}

        services::SharedPtr<T> obj;
        services::String key;
    };

    static size_t get_hash(const services::String & key)
    {
        size_t hash        = 0;
        size_t magic_p_pow = 1;
        for (size_t i = 0; i < key.length(); ++i)
        {
            hash += (static_cast<size_t>(key[i]) + 1) * magic_p_pow;
            magic_p_pow *= magic_p;
        }
        return hash;
    }

    size_t get_unique_hash(const services::String & key, services::Status & status)
    {
        size_t id             = get_hash(key) % size;
        const size_t start_id = id;
        while (_arr[id] != nullptr && _arr[id]->key != key)
        {
            id = (id + 1) % size;
            if (id == start_id)
            {
                status.add(services::ErrorHashTableCollision);
                return 0;
            }
        }
        return id;
    }

    Entry * _arr[size];
    const static size_t magic_p = 5381;
};

/** @} */
} //namespace internal
} //namespace services
} //namespace daal

#endif //__DAAL_ONEAPI_INTERNAL_HASH_TABLE_H__
