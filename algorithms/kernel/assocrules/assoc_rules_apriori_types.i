/* file: assoc_rules_apriori_types.i */
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
//  Declarations of data types used in Apriori algorithm
//--
*/

#ifndef __ASSOC_RULES_APRIORI_TYPES_I__
#define __ASSOC_RULES_APRIORI_TYPES_I__

#include "service_math.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_sort.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::algorithms::internal;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{

template <CpuType cpu>
bool assocrules_memcmp(const void* ptr1, const void* ptr2, size_t num)
{
    const char *cptr1 = (const char *)ptr1;
    const char *cptr2 = (const char *)ptr2;

    for (size_t i = 0; i < num; i++)
    {
        if (cptr1[i] != cptr2[i]) return true;
    }
    return false;
}

/** \brief Structure that specifies transaction */
template <CpuType cpu>
struct assocrules_transaction
{
    assocrules_transaction() : items(NULL), size(0), is_large(false) {}

    size_t *items;              /*<! Array of transaction's items */
    size_t size;                /*<! Transaction size */
    bool is_large;              /*<! If true then transaction is persrective
                                     for "large" itemsets search */
};

/** \brief Structure that specifies transaction's item */
template <CpuType cpu>
struct assocRulesUniqueItem
{
    size_t itemID;              /*<! Item ID */
    size_t support;             /*<! Item's support */

    /** \brief Default constructor */
    assocRulesUniqueItem(size_t _itemID = 0, size_t _support = 0) :
        itemID(_itemID), support(_support) {}

    /** \brief Copy constructor */
    assocRulesUniqueItem(const assocRulesUniqueItem &other)
    {
        itemID   = other.itemID;
        support  = other.support;
    }

    /** \brief Assignment operator */
    assocRulesUniqueItem &operator=(const assocRulesUniqueItem &other)
    {
        itemID   = other.itemID;
        support  = other.support;
        return *this;
    }
};

template <CpuType cpu>
inline bool operator< (const assocRulesUniqueItem<cpu> &lhs, const assocRulesUniqueItem<cpu> &rhs)
{
    return (lhs.itemID < rhs.itemID);
}
template <CpuType cpu>
inline bool operator> (const assocRulesUniqueItem<cpu> &lhs, const assocRulesUniqueItem<cpu> &rhs)
{
    return (rhs < lhs);
}
template <CpuType cpu>
inline bool operator<=(const assocRulesUniqueItem<cpu> &lhs, const assocRulesUniqueItem<cpu> &rhs)
{
    return !(rhs < lhs);
}
template <CpuType cpu>
inline bool operator>=(const assocRulesUniqueItem<cpu> &lhs, const assocRulesUniqueItem<cpu> &rhs)
{
    return !(lhs < rhs);
}

template <CpuType cpu>
inline void swap(assocRulesUniqueItem<cpu> &lhs, assocRulesUniqueItem<cpu> &rhs)
{
    size_t tmpItemID  = lhs.itemID;
    size_t tmpSupport = lhs.support;
    bool tmpIsLarge   = lhs.is_large;
    lhs.itemID   = rhs.itemID;
    lhs.support  = rhs.support;
    lhs.is_large = rhs.is_large;
    rhs.itemID   = tmpItemID;
    rhs.support  = tmpSupport;
    rhs.is_large = tmpIsLarge;
}

template <CpuType cpu>
int compareKeyAndUniqueItem(const void *a, const void *b)
{
    size_t key = *(size_t *)a;
    assocRulesUniqueItem<cpu> *itemB = (assocRulesUniqueItem<cpu> *)b;

    if (key < itemB->itemID) { return -1; }
    if (itemB->itemID < key) { return  1; }
    return 0;
}

/**
 *  \brief Structure describing association rule
 */
template<CpuType cpu>
struct AssocRule
{
    AssocRule(assocrules_itemset<cpu> *_left = NULL, assocrules_itemset<cpu> *_right = NULL,
              double _confidence = 0.0) :
        left(_left), right(_right), confidence(_confidence) {}

    void update(assocrules_itemset<cpu> *_left, assocrules_itemset<cpu> *_right, double _confidence = 0.0)
    {
        left       = _left;
        right      = _right;
        confidence = _confidence;
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

    assocrules_itemset<cpu> *left;       /*<! Left  part of implication */
    assocrules_itemset<cpu> *right;      /*<! Right part of implication */
    double confidence;                   /*<! Rule's confidence */
};

template <CpuType cpu>
int compareItemsetsBySupport(const void *a, const void *b)
{
    const assocrules_itemset<cpu> *aa = *((assocrules_itemset<cpu> **)a);
    const assocrules_itemset<cpu> *bb = *((assocrules_itemset<cpu> **)b);

    if (aa->support.get() < bb->support.get()) { return  1; }
    if (bb->support.get() < aa->support.get()) { return -1; }
    return 0;
}

template <CpuType cpu>
int compareRulesByConfidence(const void *a, const void *b)
{
    const AssocRule<cpu> *aa = *((AssocRule<cpu> **)a);
    const AssocRule<cpu> *bb = *((AssocRule<cpu> **)b);

    if (aa->confidence < bb->confidence) { return  1; }
    if (bb->confidence < aa->confidence) { return -1; }
    return 0;
}

/**
 *  \brief Structure that specifies input data set - a set of transactions
 */
template <CpuType cpu>
struct assocrules_dataset
{
    size_t getMaxElement(int *elementsArray, size_t nElements)
    {
        int max = 0;
        for(int i = 0; i < nElements; i++)
        {
            if (max < elementsArray[i])
            {
                max = elementsArray[i];
            }
        }
        return (size_t)max;
    }

    /** \brief Construct data set from numeric table */
    assocrules_dataset(NumericTable *dataTable, size_t _numOfTransactions, size_t _numOfUniqueItems, double minSupport)
    {
        size_t data_len = dataTable->getNumberOfRows();
        size_t itemsFullNumber;
        int *transactionID, *itemID;

        FeatureMicroTable<int, readOnly, cpu> mtTransactionID(dataTable);
        FeatureMicroTable<int, readOnly, cpu> mtItemID(dataTable);
        mtTransactionID.getBlockOfColumnValues(0, 0, data_len, &transactionID);
        mtItemID.getBlockOfColumnValues(1, 0, data_len, &itemID);

        numOfTransactions = _numOfTransactions;
        if(numOfTransactions == 0)
        {
            numOfTransactions = getMaxElement(transactionID, data_len) + 1;
        }

        itemsFullNumber = _numOfUniqueItems;
        if(itemsFullNumber == 0)
        {
            itemsFullNumber = getMaxElement(itemID, data_len) + 1;
        }

        size_t *supportVals = (size_t *)daal::services::internal::service_calloc<size_t, cpu>(itemsFullNumber);
        for (size_t i = 0; i < data_len; i++)
        {
            supportVals[itemID[i]]++;
        }
        numOfUniqueItems = 0;
        size_t iMinSupport = (size_t)daal::internal::Math<double,cpu>::sCeil(minSupport * numOfTransactions);
        for (size_t i = 0; i < itemsFullNumber; i++)
        {
            if (supportVals[i] >= iMinSupport) { numOfUniqueItems++; }
        }
        uniq_items = new assocRulesUniqueItem<cpu>[numOfUniqueItems];
        numOfUniqueItems = 0;
        for (size_t i = 0; i < itemsFullNumber; i++)
        {
            if (supportVals[i] >= iMinSupport)
            {
                uniq_items[numOfUniqueItems++] = assocRulesUniqueItem<cpu>(i, supportVals[i]);
            }
        }

        numOfLargeTransactions = 0;
        tran = new assocrules_transaction<cpu>[numOfTransactions];
        large_tran = new assocrules_transaction<cpu> *[numOfTransactions];

        size_t numItems = 0;
        size_t *items = (size_t *)daal::services::daal_malloc(numOfUniqueItems * sizeof(size_t));

        for (size_t i = 0; i < data_len; i++)
        {
            if (supportVals[itemID[i]] >= iMinSupport)
            {
                items[numItems++] = itemID[i];
            }
            if (((i <  data_len-1) && (transactionID[i + 1] != transactionID[i])) ||
                 (i == data_len-1))
            {
                if (numItems > 1)
                {
                    qSort<size_t, cpu>(numItems, items);
                    tran[numOfLargeTransactions].size = numItems;
                    tran[numOfLargeTransactions].items = (size_t *)daal::services::daal_malloc(numItems * sizeof(size_t));
                    tran[numOfLargeTransactions].is_large = true;
                    daal::services::daal_memcpy_s(tran[numOfLargeTransactions].items, numItems * sizeof(size_t), items, numItems * sizeof(size_t));
                    large_tran[numOfLargeTransactions] = &tran[numOfLargeTransactions];
                    numOfLargeTransactions++;
                }
                numItems = 0;
            }
        }

        daal::services::daal_free(items);
        daal::services::daal_free(supportVals);
        mtTransactionID.release();
        mtItemID.release();
    }

    ~assocrules_dataset()
    {
        for (size_t i = 0; i < numOfTransactions; i++)
        {
            daal::services::daal_free(tran[i].items);
        }
        delete[] tran;
        delete[] large_tran;
        delete[] uniq_items;
    }

    assocrules_transaction<cpu> *tran;                      /*<! Array of transactions */
    size_t numOfTransactions;                               /*<! Number of transactions */
    assocrules_transaction<cpu> **large_tran;               /*<! Array of pointers to "large" transactions
                                                                 that are perspective for "large" itemsets search */
    size_t numOfLargeTransactions;                          /*<! Number of "large" transactions */
    assocRulesUniqueItem<cpu> *uniq_items;                  /*<! Array of unique items */
    size_t numOfUniqueItems;                                /*<! Number of unique items */

private:
    assocrules_dataset(const assocrules_dataset &) {};
};

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
