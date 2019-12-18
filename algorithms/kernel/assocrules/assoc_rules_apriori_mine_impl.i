/* file: assoc_rules_apriori_mine_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of auxiliary functions for association rules
//  Apriori method.
//--
*/

#ifndef __ASSOC_RULES_APRIORI_MINE_IMPL_I__
#define __ASSOC_RULES_APRIORI_MINE_IMPL_I__

#include "service_memory.h"
#include "service_math.h"
#include "service_sort.h"

#include "threading.h"
#include "assoc_rules_apriori_types.i"
#include "assoc_rules_apriori_tree.i"

using namespace daal::algorithms::internal;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{
/**
 *  \brief Find "large" itemsets of size 1
 *
 *  \param imin_s[in]    minimum support
 *  \param data[in]      input data set
 *  \param L[out]        structure containing "large" itemsets
 *  \return Status object
 */
template <typename algorithmFPType, CpuType cpu>
services::Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::firstPass(size_t imin_s, assocrules_dataset<cpu> & data, ItemSetList<cpu> & l)
{
    size_t numOfUniqueItems                 = data.numOfUniqueItems;
    assocRulesUniqueItem<cpu> * uniqueItems = data.uniq_items;
    /* Form "large" item sets of size 1 from unique items which support
       is greater than minimum support */
    for (size_t i = 0; i < numOfUniqueItems; i++)
    {
        size_t item_id               = uniqueItems[i].itemID;
        size_t item_support          = uniqueItems[i].support;
        assocrules_itemset<cpu> * ai = new assocrules_itemset<cpu>(item_id, item_support);
        DAAL_CHECK_MALLOC(ai);
        DAAL_CHECK_STATUS_OK(ai->ok(), ai->getLastStatus());
        l.insert(ai);
    }
    return (numOfUniqueItems > 0) ? services::Status() : services::ErrorAprioriIncorrectInputData;
}

/**
 *  \brief Test that all {n-1}-item subsets of {n}-item set are "large" item sets
 *
 *  \param iset_size[in] length of the item set - {n}
 *  \param candidate[in] item set of length {n} to test
 *  \param subset[in]    auxiliary buffer of length {n-1}
 *  \param L_prev[in]    list of "large" item sets of length {n-1}
 *
 *  \return
 *  false - if all {n-1}-item subsets of the "candidate" are "large"
 *  true  - otherwise
 */
template <typename algorithmFPType, CpuType cpu>
bool AssociationRulesKernel<apriori, algorithmFPType, cpu>::pruneCandidate(size_t iset_size, const size_t * candidate, size_t * subset,
                                                                           hash_tree<cpu> & C_tree)
{
    assocrules_itemset<cpu> * iset = nullptr;
    int levelMiss                  = 0;
    for (size_t i = 1; i < iset_size; i++)
    {
        for (size_t k = 0; k < i; k++)
        {
            subset[k] = candidate[k];
        }
        for (size_t k = i + 1; k < iset_size; k++)
        {
            subset[k - 1] = candidate[k];
        }

        iset = C_tree.hash_subset(iset_size - 1, subset, &levelMiss);
        if (!iset)
        {
            return true;
        }
    }

    return false;
}

template <typename algorithmFPType, CpuType cpu>
assocrules_itemset<cpu> * AssociationRulesKernel<apriori, algorithmFPType, cpu>::genCandidate(size_t iset_size, size_t * first_items,
                                                                                              size_t second_item, size_t * subset_buf,
                                                                                              hash_tree<cpu> * C_tree, services::Status & s)
{
    s = services::Status();
    /* Create candidate itemset */
    assocrules_itemset<cpu> * iset = new assocrules_itemset<cpu>(iset_size + 1, first_items, second_item);

    if (!iset)
    {
        s = services::ErrorMemoryAllocationFailed;
        return nullptr;
    }

    if (!iset->ok())
    {
        s = iset->getLastStatus();
        delete iset;
        iset = nullptr;
        return nullptr;
    }

    /* Check apriori property for candidate itemset */
    if (pruneCandidate(iset_size + 1, iset->items, subset_buf, *C_tree))
    {
        delete iset;
        iset = nullptr;
        return nullptr;
    }
    return iset;
}

template <typename algorithmFPType, CpuType cpu>
size_t AssociationRulesKernel<apriori, algorithmFPType, cpu>::binarySearch(size_t nUniqueItems, assocRulesUniqueItem<cpu> * uniqueItems,
                                                                           size_t itemID)
{
    size_t lo = 0;
    size_t hi = nUniqueItems - 1;
    size_t me = ((hi + lo) >> 1);
    while (lo < hi)
    {
        if (uniqueItems[me].itemID < itemID)
        {
            lo = me + 1;
        }
        else if (itemID < uniqueItems[me].itemID)
        {
            hi = me - 1;
        }
        else
        {
            return me;
        }
        me = ((hi + lo) >> 1);
    }
    return me;
}

/**
 *  \brief Generate candidate itemsets of size iset_size+1
 *         from "large" itemsets of size iset_size
 *
 *  \param iset_size[in]  length of input itemsets
 *  \param L[in]          structure containing "large" itemsets
 *  \return false if no candidates were generated, true otherwise
 *  \param outputStatus[out]    Status object that indicates tehe result of memory allocation
 */
template <typename algorithmFPType, CpuType cpu>
bool AssociationRulesKernel<apriori, algorithmFPType, cpu>::genCandidates(size_t iset_size, ItemSetList<cpu> * L, hash_tree<cpu> * C_tree,
                                                                          size_t nUniqueItems, assocRulesUniqueItem<cpu> * uniqueItems,
                                                                          services::Status & outputStatus)
{
    outputStatus              = services::Status();
    size_t new_iset_size      = iset_size + 1;
    ItemSetList<cpu> * L_prev = &(L[iset_size - 1]);
    ItemSetList<cpu> * L_cur  = &(L[iset_size]);
    TArray<size_t, cpu> subset_ar(iset_size);
    size_t * subset_buf = subset_ar.get();
    if (!subset_buf) return false;

    if (iset_size == 1 && L[0].size > 1)
    {
        /* Here if candidates of size 2 are generated */
        /* Generate all combinations of size 2 from "large" itemsets of size 1 */

        for (size_t i = 0; i < nUniqueItems; i++)
        {
            for (size_t j = i + 1; j < nUniqueItems; j++)
            {
                services::Status s;
                assocrules_itemset<cpu> * ai = new assocrules_itemset<cpu>(new_iset_size, &(uniqueItems[i].itemID), uniqueItems[j].itemID);

                if (!ai)
                {
                    outputStatus = services::ErrorMemoryAllocationFailed;
                    return false;
                }

                if (!ai->ok())
                {
                    outputStatus = ai->getLastStatus();
                    delete ai;
                    ai = nullptr;
                    return false;
                }

                L_cur->insert(ai);
            }
        }
    }
    else
    {
        /* Here if candidates of size greater than 2 are generated */
        assocrules_itemset<cpu> * iset = nullptr;
        for (auto first_it = L_prev->start; first_it; first_it = first_it->next())
        {
            auto first_items     = first_it->itemSet()->items;
            size_t last_item     = first_items[iset_size - 1];
            size_t start_item_id = binarySearch(nUniqueItems, uniqueItems, last_item);
            for (size_t k = start_item_id; k < nUniqueItems; k++)
            {
                services::Status s;
                iset = genCandidate(iset_size, first_items, uniqueItems[k].itemID, subset_buf, C_tree, s);

                if (!s.ok())
                {
                    outputStatus = s;
                    return false;
                }

                if (iset)
                {
                    L_cur->insert(iset);
                }
            }
        }
    }
    return (L_cur->size > 0);
}

/**
 *  \brief Generate all subsets of size iset_size from a transaction
 *         and hash those subsets using hash tree of candidate itemsets C_tree
 *         to increment support values of candidates.
 *
 *  \param items[in]        transaction items
 *  \param iset_size[in]    size of candidate itemsets
 *  \param subset[in]       buffer for storing transaction items subsets
 *  \param idx[in]          buffer for storing transaction items indices
 *  \param C_tree[in]       hash tree formed from candidates
 *  \param large_count[out] number of candidates found in transaction
 */
template <typename algorithmFPType, CpuType cpu>
void AssociationRulesKernel<apriori, algorithmFPType, cpu>::genSubset(size_t transactionSize, const size_t * items, size_t iset_size, size_t * subset,
                                                                      size_t * idx, hash_tree<cpu> & C_tree, size_t & large_count)
{
    int levelMiss;
    assocrules_itemset<cpu> * iset = nullptr;
    large_count                    = 0;

    size_t h = iset_size;
    idx[0]   = -1;
    do
    {
        /* Generate next subset of size iset_size */
        idx[iset_size - h]++;
        for (size_t j = iset_size - h + 1; j < iset_size; j++)
        {
            idx[j] = idx[j - 1] + 1;
        }
        for (size_t j = 0; j < iset_size; j++)
        {
            subset[j] = items[idx[j]];
        }

        /* Increment counter for the subset if this subset exists
           in hash tree C_tree */
        iset = C_tree.hash_subset(iset_size, subset, &levelMiss);
        if (iset)
        {
            iset->support.inc();
            large_count++;
        }

        if ((levelMiss > 0) && (idx[levelMiss - 1] < transactionSize - 1))
        {
            h = iset_size - (levelMiss - 1);
            while (h <= iset_size && idx[iset_size - h] >= transactionSize - h)
            {
                h++;
            }
        }
        else if (idx[iset_size - h] < transactionSize - h)
        {
            h = 1;
        }
        else
        {
            h++;
        }
    } while (idx[0] < transactionSize - iset_size);
}

template <CpuType cpu>
void removeNodesWithSmallSupport(ItemSetList<cpu> & l, size_t support)
{
    for (typename ItemSetList<cpu>::Node *cur = l.start, *prev = nullptr; cur;)
    {
        if (cur->itemSet()->support.get() < support)
        {
            auto next = cur->next();
            l.removeNode(cur, prev);
            cur = next;
            //prev remains the same
        }
        else
        {
            prev = cur;
            cur  = cur->next();
        }
    }
}

/**
 *  \brief Remove candidate itemsets which support is less then minimum support
 *         from array of "large" item sets
 *
 *  \param imin_s[in]    minimum support
 *  \param iset_size[in] size (number of items) of the candidate itemsets
 *  \param data[in]      input data set
 *  \param L[in]         structure containing "large" item sets
 *
 */
template <typename algorithmFPType, CpuType cpu>
void AssociationRulesKernel<apriori, algorithmFPType, cpu>::prune(size_t imin_s, size_t iset_size, assocrules_dataset<cpu> & data,
                                                                  ItemSetList<cpu> * L, hash_tree<cpu> * C_tree)
{
    size_t new_iset_size = iset_size + 1;
    daal::tls<size_t *> tls([&]() -> size_t * // The functor to initialize a memory buffer
                            { return daal::services::internal::service_calloc<size_t, cpu>(2 * new_iset_size); });

    assocrules_transaction<cpu> ** large_tran = data.large_tran;
    size_t numOfLargeTransactions             = data.numOfLargeTransactions;

    daal::threader_for(numOfLargeTransactions, numOfLargeTransactions, [=, &tls](size_t i_tran) {
        assocrules_transaction<cpu> * tran = large_tran[i_tran];

        size_t large_count = 0;
        size_t * idx       = tls.local();
        size_t * subset    = idx + new_iset_size;

        genSubset(tran->size, tran->items, new_iset_size, subset, idx, *C_tree, large_count);

        if (large_count < 2)
        {
            tran->is_large = false;
        }
    });

    tls.reduce([&](size_t * idx) {
        daal::services::daal_free(idx);
        idx = nullptr;
    });

    /* Remove candidates that has support less than mininmum support from hash tree */
    for (size_t i = 0; i < C_tree->n_leaves; i++)
    {
        ItemSetList<cpu> & isetList = C_tree->leaves[i].iset_list;
        removeNodesWithSmallSupport<cpu>(isetList, imin_s);
    }

    /* Remove candidates that has support less than mininmum support */
    removeNodesWithSmallSupport<cpu>(L[iset_size], imin_s);

    /* Remove transactions that are not perspective for large itemsets search */
    size_t iLarge    = 0;
    size_t iNotLarge = numOfLargeTransactions - 1;
    while (iLarge < iNotLarge && iLarge < numOfLargeTransactions)
    {
        while (iLarge < numOfLargeTransactions && large_tran[iLarge]->is_large)
        {
            iLarge++;
        }
        while (iNotLarge > iLarge && !large_tran[iNotLarge]->is_large)
        {
            iNotLarge--;
        }

        if (iLarge >= iNotLarge || iLarge >= numOfLargeTransactions || iNotLarge <= 0)
        {
            break;
        }

        assocrules_transaction<cpu> * tmp = large_tran[iLarge];
        large_tran[iLarge]                = large_tran[iNotLarge];
        large_tran[iNotLarge]             = tmp;
        iLarge++;
        iNotLarge--;
    }
    data.numOfLargeTransactions = iLarge;
}

/**
 *  \brief Generate "large" item sets of size k+1 from "large" item sets of size k.
 *
 *  \param imin_s[in]      minimum support
 *  \param iset_size[in]   size of the "large" item sets generated on previous stage (k)
 *  \param data[in]        input data set
 *  \param L[out]          structure containing "large" item sets
 *  \param L_size[out]     size of the array L
 *  \param bFound[out]  flag. true,  if at least 2 "large" item sets of size k+1 were found;
 *                               false, otherwise
 *  \param s[out]       Status object that indicates the result of memory allocation
 */
template <typename algorithmFPType, CpuType cpu>
hash_tree<cpu> * AssociationRulesKernel<apriori, algorithmFPType, cpu>::nextPass(size_t imin_s, size_t iset_size, assocrules_dataset<cpu> & data,
                                                                                 ItemSetList<cpu> * L, size_t & L_size, bool & bFound,
                                                                                 hash_tree<cpu> * C_tree, services::Status & s)
{
    s      = services::Status();
    bFound = genCandidates(iset_size, L, C_tree, data.numOfUniqueItems, data.uniq_items, s);

    if (!s.ok()) return nullptr;

    delete C_tree;
    C_tree = nullptr;
    if (!bFound) return nullptr;
    hash_tree<cpu> * C_tree_new = new hash_tree<cpu>(iset_size + 1, L[iset_size]);
    if (!C_tree_new)
    {
        bFound = false;
        s      = services::ErrorMemoryAllocationFailed;
        return nullptr;
    }
    if (!C_tree_new->ok())
    {
        bFound = false;
        s      = C_tree_new->getLastStatus();
        delete C_tree_new;
        C_tree_new = nullptr;
        return nullptr;
    }

    prune(imin_s, iset_size, data, L, C_tree_new);
    if (L[iset_size].size > 0)
    {
        ++L_size;
    }
    if (L[iset_size].size < 2)
    {
        bFound = false;
    }
    return C_tree_new;
}

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
