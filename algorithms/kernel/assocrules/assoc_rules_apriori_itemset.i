/* file: assoc_rules_apriori_itemset.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Definition of data strustures describing "large" item sets.
//--
*/

#ifndef __ASSOC_RULES_APRIORI_ITEMSET_I__
#define __ASSOC_RULES_APRIORI_ITEMSET_I__

#include "service_memory.h"
#include "daal_atomic_int.h"

using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{

/**
 *  \brief Structure describing itemset
 */
template<CpuType cpu>
struct assocrules_itemset
{
    /** \brief Construct itemset of size 1 from item value */
    assocrules_itemset(size_t item0, size_t _support = 0) : size(1), support(_support)
    {
        items = new size_t[1];
        items[0] = item0;
    }

    /**
     *  \brief Construct itemset of size (iset_size) from itemset of size (iset_size - 1) and item
     */
    assocrules_itemset(const size_t iset_size, const size_t *first_items,
                       const size_t second_item, const size_t _support = 0) :
        support(_support), size(iset_size)
    {
        items = new size_t[iset_size];

        daal::services::daal_memcpy_s(items, iset_size*sizeof(size_t), first_items, (iset_size - 1)*sizeof(size_t));
        items[iset_size - 1] = second_item;
    }

    ~assocrules_itemset()
    {
        delete [] items;
        items = NULL;
        size = 0;
    }

    /** \brief Copy constructor */
    assocrules_itemset(const assocrules_itemset &iset)
    {
        size = iset.size;
        support.set(iset.support.get());
        items = new size_t[size];
        daal::services::daal_memcpy_s(items, size * sizeof(size_t), iset.items, size * sizeof(size_t));
    }

    assocrules_itemset &operator=(const assocrules_itemset &iset)
    {
        if (this != &iset)
        {
            delete [] items;
            size = iset.size;
            support.set(iset.support.get());
            items = new size_t[size];
            daal::services::daal_memcpy_s(items, size * sizeof(size_t), iset.items, size * sizeof(size_t));
        }
        return *this;
    }

    Atomic<size_t> support;
    size_t *items;              /*<! Array of items */
    size_t size;                /*<! Itemset size */
};

/** \brief Structure describing an itemset list */
template<CpuType cpu>
struct ItemSetList
{
    struct Node
    {
        Node() : next(NULL), itemSet(NULL) {}
        Node *next;
        assocrules_itemset<cpu> *itemSet;
    };

    /* Create list of zero length */
    ItemSetList() : start(NULL), end(NULL), current(NULL), size(0), errors(new services::KernelErrorCollection()) {}

    void remove()
    {
        if (start == NULL) { return; }
        while (start->next)
        {
            current = start->next;
            delete start->itemSet;
            delete start;
            start = current;
        }
        delete start->itemSet;
        delete start;
        start = NULL;
    }

    /* Release memory associated with list without deleting item sets
       associated with this list */
    virtual ~ItemSetList()
    {
        if (start == NULL) { return; }
        while (start->next)
        {
            current = start->next;
            delete start;
            start = current;
        }
        delete start;
        start = NULL;
    }

    /* Add new Node to the end of the list */
    void insert(assocrules_itemset<cpu> *itemSet)
    {
        Node *newNode = new Node;
        if (!newNode) { errors->add(services::ErrorMemoryAllocationFailed); return; }

        newNode->itemSet = itemSet;
        if (size > 0) { end->next = newNode; }
        else { start = newNode; }

        end = newNode;
        newNode->next = NULL;
        size++;
    }

    /* Removes current Node and its content */
    void removeCurrentNode()
    {
        if (!current) { return; }
        if (current == start)
        {
            start = start->next;
            delete current->itemSet;
            delete current;
            current = start;
            size--;
            return;
        }

        /* Pointer to the Node preceding 'current' */
        Node *prevNode = start;
        while (prevNode->next != current)
        {
            prevNode = prevNode->next;
        }

        prevNode->next = current->next;
        if (current->next == NULL)
        {
            end = prevNode;
        }
        delete current->itemSet;
        delete current;
        current = prevNode->next;
        size--;
    }

    /* Removes current Node but doesn't remove its content */
    void excludeCurrentNode()
    {
        if (!current) { return; }
        if (current == start)
        {
            start = start->next;
            delete current;
            current = start;
            size--;
            return;
        }

        /* Pointer to the Node preceding 'current' */
        Node *prevNode = start;
        while (prevNode->next != current)
        {
            prevNode = prevNode->next;
        }

        prevNode->next = current->next;
        if (current->next == NULL)
        {
            end = prevNode;
        }
        delete current;
        current = prevNode->next;
        size--;
    }

    static void *operator new(size_t sz)
    {
        return daal::services::daal_malloc(sz);
    }
    static void *operator new[](size_t sz)
    {
        return daal::services::daal_malloc(sz);
    }

    static void operator delete(void *ptr, size_t sz)
    {
        daal::services::daal_free(ptr);
    }
    static void operator delete[](void *ptr, size_t sz)
    {
        daal::services::daal_free(ptr);
    }

    Node *start;
    Node *end;
    Node *current;
    size_t size;
    services::SharedPtr<services::KernelErrorCollection> errors;
};

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
