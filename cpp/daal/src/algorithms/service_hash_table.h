/* file: service_hash_table.h */
/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef __SERVICE_HASH_TABLE_H__
#define __SERVICE_HASH_TABLE_H__

#include "src/services/service_utils.h"

namespace daal
{
namespace algorithms
{
namespace internal
{
using namespace services::internal;

template <CpuType cpu, typename KeyType>
struct Hash
{
    size_t operator()(KeyType a) const;
};

// Robert Jenkins' 32 bit integer hash function
template <CpuType cpu>
struct Hash<cpu, uint32_t>
{
    size_t operator()(uint32_t a) const
    {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return static_cast<size_t>(a);
    }
};

template <CpuType cpu, typename KeyType, typename ValueType, typename HashFuncType = Hash<cpu, KeyType> >
class HashTable
{
public:
    HashTable(const size_t size) : _size(size), _table(size)
    {
        for (size_t i = 0; i < _size; ++i)
        {
            _table[i].next   = nullptr;
            _table[i].isFree = true;
        }
    }

    ~HashTable()
    {
        for (size_t i = 0; i < _size; ++i)
        {
            Entry * curr = _table[i].next;
            while (curr)
            {
                Entry * next = curr->next;
                delete curr;
                curr = next;
            }
        }
    }

    bool find(const KeyType & key, ValueType & value)
    {
        const size_t id = _hashFunc(key) % _size;
        if (_table[id].isFree)
        {
            return false;
        }
        else if (_table[id].key == key)
        {
            value = _table[id].value;
            return true;
        }
        else
        {
            Entry * entry = _table[id].next;
            while (entry)
            {
                if (entry->key == key)
                {
                    value = entry->value;
                    return true;
                }
                entry = entry->next;
            }
        }
        return false;
    }

    void insert(const KeyType & key, const ValueType & value)
    {
        const size_t id = _hashFunc(key) % _size;
        if (_table[id].isFree)
        {
            _table[id].isFree = false;
            _table[id].key    = key;
            _table[id].value  = value;
        }
        else
        {
            Entry * entry = _table[id].next;
            Entry * prev  = nullptr;
            while (entry && entry->key != key)
            {
                prev  = entry;
                entry = entry->next;
            }
            if (entry)
            {
                entry->value = value;
            }
            else
            {
                entry = Entry::create(key, value);
                if (!entry) return;
                if (prev)
                {
                    prev->next = entry;
                }
                else
                {
                    _table[id].next = entry;
                }
            }
        }
    }

    void erase(const KeyType & key)
    {
        const size_t id = _hashFunc(key) % _size;
        Entry * entry   = _table[id].next;
        if (_table[id].isFree) return;

        if (_table[id].key == key)
        {
            if (entry)
            {
                _table[id].key   = entry->key;
                _table[id].value = entry->value;
                _table[id].next  = entry->next;
                delete entry;
            }
            else
            {
                _table[id].isFree = true;
            }
        }
        else
        {
            Entry * prev = nullptr;
            while (entry && entry->key != key)
            {
                prev  = entry;
                entry = entry->next;
            }
            if (entry)
            {
                if (prev)
                {
                    prev->next = entry->next;
                }
                else
                {
                    _table[id].next = entry->next;
                }
                delete entry;
            }
        }
    }

private:
    struct Entry
    {
        DAAL_NEW_DELETE();

        static Entry * create(const KeyType & key, const ValueType & value)
        {
            auto val = new Entry(key, value);
            if (val) return val;
            delete val;
            return nullptr;
        }

        KeyType key;
        ValueType value;
        Entry * next;

    private:
        Entry(const KeyType & keyIn, const ValueType & valueIn) : key(keyIn), value(valueIn), next(nullptr) {}
    };

    struct FirstEntry
    {
        KeyType key;
        ValueType value;
        Entry * next;
        bool isFree;
    };

    const size_t _size;
    TArray<FirstEntry, cpu> _table;
    HashFuncType _hashFunc;
};

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif //__SERVICE_HASH_MAP_H__
