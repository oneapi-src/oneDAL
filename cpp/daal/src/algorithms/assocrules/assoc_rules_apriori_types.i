/* file: assoc_rules_apriori_types.i */
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
//  Declarations of data types used in Apriori algorithm
//--
*/

#ifndef __ASSOC_RULES_APRIORI_TYPES_I__
#define __ASSOC_RULES_APRIORI_TYPES_I__

#include "src/externals/service_math.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_sort.h"

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
bool assocrules_memcmp(const size_t * ptr1, const size_t * ptr2, size_t num)
{
    size_t i = 0;
    for (; (i < num) && (ptr1[i] == ptr2[i]); ++i)
        ;
    return i != num;
}

/** \brief Structure that specifies transaction */
template <CpuType cpu>
struct assocrules_transaction
{
    DAAL_NEW_DELETE();
    assocrules_transaction() : items(NULL), size(0), is_large(false) {}

    size_t * items; /*<! Array of transaction's items */
    size_t size;    /*<! Transaction size */
    bool is_large;  /*<! If true then transaction is persrective
                                     for "large" itemsets search */
};

/** \brief Structure that specifies transaction's item */
template <CpuType cpu>
struct assocRulesUniqueItem
{
    DAAL_NEW_DELETE();
    size_t itemID;  /*<! Item ID */
    size_t support; /*<! Item's support */

    /** \brief Default constructor */
    assocRulesUniqueItem(size_t _itemID = 0, size_t _support = 0) : itemID(_itemID), support(_support) {}

    /** \brief Copy constructor */
    assocRulesUniqueItem(const assocRulesUniqueItem & other)
    {
        itemID  = other.itemID;
        support = other.support;
    }

    /** \brief Assignment operator */
    assocRulesUniqueItem & operator=(const assocRulesUniqueItem & other)
    {
        itemID  = other.itemID;
        support = other.support;
        return *this;
    }
};

template <CpuType cpu>
inline bool operator<(const assocRulesUniqueItem<cpu> & lhs, const assocRulesUniqueItem<cpu> & rhs)
{
    return (lhs.itemID < rhs.itemID);
}
template <CpuType cpu>
inline bool operator>(const assocRulesUniqueItem<cpu> & lhs, const assocRulesUniqueItem<cpu> & rhs)
{
    return (rhs < lhs);
}
template <CpuType cpu>
inline bool operator<=(const assocRulesUniqueItem<cpu> & lhs, const assocRulesUniqueItem<cpu> & rhs)
{
    return !(rhs < lhs);
}
template <CpuType cpu>
inline bool operator>=(const assocRulesUniqueItem<cpu> & lhs, const assocRulesUniqueItem<cpu> & rhs)
{
    return !(lhs < rhs);
}

template <CpuType cpu>
inline void swap(assocRulesUniqueItem<cpu> & lhs, assocRulesUniqueItem<cpu> & rhs)
{
    size_t tmpItemID  = lhs.itemID;
    size_t tmpSupport = lhs.support;
    bool tmpIsLarge   = lhs.is_large;
    lhs.itemID        = rhs.itemID;
    lhs.support       = rhs.support;
    lhs.is_large      = rhs.is_large;
    rhs.itemID        = tmpItemID;
    rhs.support       = tmpSupport;
    rhs.is_large      = tmpIsLarge;
}

template <CpuType cpu>
int compareKeyAndUniqueItem(const void * a, const void * b)
{
    size_t key                        = *(size_t *)a;
    assocRulesUniqueItem<cpu> * itemB = (assocRulesUniqueItem<cpu> *)b;

    if (key < itemB->itemID)
    {
        return -1;
    }
    if (itemB->itemID < key)
    {
        return 1;
    }
    return 0;
}

/**
 *  \brief Structure describing association rule
 */
template <CpuType cpu>
struct AssocRule
{
    DAAL_NEW_DELETE();
    AssocRule(const assocrules_itemset<cpu> * _left = NULL, const assocrules_itemset<cpu> * _right = NULL, double _confidence = 0.0)
        : left(_left), right(_right), confidence(_confidence)
    {}

    void update(const assocrules_itemset<cpu> * _left, const assocrules_itemset<cpu> * _right, double _confidence = 0.0)
    {
        left       = _left;
        right      = _right;
        confidence = _confidence;
    }

    const assocrules_itemset<cpu> * left;  /*<! Left  part of implication */
    const assocrules_itemset<cpu> * right; /*<! Right part of implication */
    double confidence;                     /*<! Rule's confidence */
};

template <CpuType cpu>
int compareItemsetsBySupport(const void * a, const void * b)
{
    typedef const assocrules_itemset<cpu> * ItemsetConstPtr;
    ItemsetConstPtr aa = *((ItemsetConstPtr *)a);
    ItemsetConstPtr bb = *((ItemsetConstPtr *)b);

    if (bb->support.get() < aa->support.get())
    {
        return -1;
    }
    return (aa->support.get() < bb->support.get()) ? 1 : 0;
}

template <CpuType cpu>
int compareRulesByConfidence(const void * a, const void * b)
{
    const AssocRule<cpu> * aa = *((AssocRule<cpu> **)a);
    const AssocRule<cpu> * bb = *((AssocRule<cpu> **)b);

    if (bb->confidence < aa->confidence)
    {
        return -1;
    }
    return (aa->confidence < bb->confidence) ? 1 : 0;
}

/**
 *  \brief Structure that specifies input data set - a set of transactions
 */
template <CpuType cpu>
struct assocrules_dataset
{
    size_t getMaxElement(const int * elementsArray, size_t nElements) const
    {
        int max = 0;
        for (size_t i = 0; i < nElements; i++)
        {
            if (max < elementsArray[i])
            {
                max = elementsArray[i];
            }
        }
        DAAL_ASSERT(max >= 0)
        return (size_t)max;
    }

    /** \brief Construct data set from numeric table */
    assocrules_dataset(NumericTable * dataTable, size_t _numOfTransactions, size_t _numOfUniqueItems, double minSupport)
        : tran(nullptr), large_tran(nullptr), uniq_items(nullptr), numOfTransactions(0)
    {
        _status         = services::Status();
        size_t data_len = dataTable->getNumberOfRows();

        ReadColumns<int, cpu> mtTransactionID(dataTable, 0, 0, data_len);
        ReadColumns<int, cpu> mtItemID(dataTable, 1, 0, data_len);
        const int * transactionID = mtTransactionID.get();
        const int * itemID        = mtItemID.get();
        if (!(transactionID && itemID)) return;
        numOfTransactions = _numOfTransactions;
        if (numOfTransactions == 0)
        {
            numOfTransactions = getMaxElement(transactionID, data_len) + 1;
        }

        size_t itemsFullNumber = _numOfUniqueItems;
        if (itemsFullNumber == 0)
        {
            itemsFullNumber = getMaxElement(itemID, data_len) + 1;
        }

        size_t * supportVals = (size_t *)daal::services::internal::service_calloc<size_t, cpu>(itemsFullNumber);

        if (!supportVals)
        {
            _status = services::ErrorMemoryAllocationFailed;
            return;
        }

        for (size_t i = 0; i < data_len; i++)
        {
            supportVals[itemID[i]]++;
        }
        numOfUniqueItems = 0;
        double ceil      = daal::internal::MathInst<double, cpu>::sCeil(minSupport * numOfTransactions);
        DAAL_ASSERT(ceil >= 0)

        size_t iMinSupport = (size_t)ceil;
        for (size_t i = 0; i < itemsFullNumber; i++)
        {
            if (supportVals[i] >= iMinSupport)
            {
                numOfUniqueItems++;
            }
        }
        uniq_items = new assocRulesUniqueItem<cpu>[numOfUniqueItems];

        if (!uniq_items)
        {
            daal::services::daal_free(supportVals);
            supportVals = nullptr;

            _status = services::ErrorMemoryAllocationFailed;
            return;
        }

        numOfUniqueItems = 0;
        for (size_t i = 0; i < itemsFullNumber; i++)
        {
            if (supportVals[i] >= iMinSupport)
            {
                uniq_items[numOfUniqueItems++] = assocRulesUniqueItem<cpu>(i, supportVals[i]);
            }
        }

        numOfLargeTransactions = 0;
        tran                   = new assocrules_transaction<cpu>[numOfTransactions];
        large_tran             = new assocrules_transaction<cpu> *[numOfTransactions];

        if (!tran)
        {
            daal::services::daal_free(supportVals);
            delete[] uniq_items;
            supportVals = nullptr;
            uniq_items  = nullptr;

            _status = services::ErrorMemoryAllocationFailed;
            return;
        }

        if (!large_tran)
        {
            daal::services::daal_free(supportVals);
            delete[] tran;
            delete[] uniq_items;
            supportVals = nullptr;
            tran        = nullptr;
            uniq_items  = nullptr;

            _status = services::ErrorMemoryAllocationFailed;
            return;
        }

        size_t numItems = 0;
        size_t * items  = (size_t *)daal::services::daal_malloc(numOfUniqueItems * sizeof(size_t));

        if (!items)
        {
            daal::services::daal_free(supportVals);
            delete[] tran;
            delete[] large_tran;
            delete[] uniq_items;

            supportVals = nullptr;
            tran        = nullptr;
            large_tran  = nullptr;
            uniq_items  = nullptr;

            _status = services::ErrorMemoryAllocationFailed;
            return;
        }
        int result = 0;
        for (size_t i = 0; i < data_len; i++)
        {
            if (supportVals[itemID[i]] >= iMinSupport)
            {
                items[numItems++] = itemID[i];
            }
            if (((i < data_len - 1) && (transactionID[i + 1] != transactionID[i])) || (i == data_len - 1))
            {
                if (numItems > 1)
                {
                    qSort<size_t, cpu>(numItems, items);
                    tran[numOfLargeTransactions].size     = numItems;
                    tran[numOfLargeTransactions].items    = (size_t *)daal::services::daal_malloc(numItems * sizeof(size_t));
                    tran[numOfLargeTransactions].is_large = true;
                    result |= daal::services::internal::daal_memcpy_s(tran[numOfLargeTransactions].items, numItems * sizeof(size_t), items,
                                                                      numItems * sizeof(size_t));
                    large_tran[numOfLargeTransactions] = &tran[numOfLargeTransactions];
                    numOfLargeTransactions++;
                }
                numItems = 0;
            }
        }
        if (result)
        {
            _status |= services::Status(services::ErrorMemoryCopyFailedInternal);
        }

        daal::services::daal_free(items);
        daal::services::daal_free(supportVals);
    }

    ~assocrules_dataset()
    {
        for (size_t i = 0; i < numOfTransactions; i++)
        {
            daal::services::daal_free(tran[i].items);
            tran[i].items = nullptr;
        }
        delete[] tran;
        delete[] large_tran;
        delete[] uniq_items;
        tran       = nullptr;
        large_tran = nullptr;
        uniq_items = nullptr;
    }

    assocrules_transaction<cpu> * tran;        /*<! Array of transactions */
    size_t numOfTransactions;                  /*<! Number of transactions */
    assocrules_transaction<cpu> ** large_tran; /*<! Array of pointers to "large" transactions
                                                                 that are perspective for "large" itemsets search */
    size_t numOfLargeTransactions;             /*<! Number of "large" transactions */
    assocRulesUniqueItem<cpu> * uniq_items;    /*<! Array of unique items */
    size_t numOfUniqueItems;                   /*<! Number of unique items */

    bool ok() const { return _status.ok(); }
    services::Status getLastStatus() const { return _status; }

protected:
    services::Status _status;

private:
    assocrules_dataset(const assocrules_dataset &) { _status = services::Status(); };
};

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
