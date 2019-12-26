/* file:  assoc_rules_apriori_discover_impl.i */
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
//  Implementation of rules discovering stage of association rules mining
//  algorithm.
//--
*/

#ifndef __ASSOC_RULES_APRIORI_DISCOVER_IMPL_I__
#define __ASSOC_RULES_APRIORI_DISCOVER_IMPL_I__

#include "service_memory.h"
#include "assoc_rules_apriori_types.i"

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{
/**
 *  Find item set in the list of item sets
 *
 *  \param items_size[in]   item set size
 *  \param items[in]        item set (array of items)
 *  \param L_cur[in]        list of the item sets to search for item set
 *
 *  \return Pointer to the found item set. NULL if no item set was found
 */
template <typename algorithmFPType, CpuType cpu>
const assocrules_itemset<cpu> * AssociationRulesKernel<apriori, algorithmFPType, cpu>::findItemSet(size_t items_size, const size_t * items,
                                                                                                   const ItemSetList<cpu> & L_cur)
{
    for (auto current = L_cur.start; current; current = current->next())
    {
        if (!assocrules_memcmp<cpu>(items, current->itemSet()->items, items_size))
        {
            return current->itemSet();
        }
    }
    return NULL;
}

/**
 *  Find intersection between two item sets
 *
 *  \param a[in]            array of items of the 1st item set (sorted)
 *  \param aSize[in]        number of items in the 1st item set
 *  \param b[in]            array of items of the 2nd item set (sorted)
 *  \param bSize[in]        number of items in the 2nd item set
 *  \param c[out]           array of items in the intersection of 1st and 2nd item sets
 *  \param cSize[out]    number of items in the intersection of 1st and 2nd item sets
 *
  */
template <typename algorithmFPType, CpuType cpu>
void AssociationRulesKernel<apriori, algorithmFPType, cpu>::setIntersection(const size_t * a, size_t aSize, const size_t * b, size_t bSize,
                                                                            size_t * c, size_t & cSize)
{
    cSize     = 0;
    size_t ia = 0, ib = 0;

    while (ia < aSize && ib < aSize)
    {
        if (a[ia] == b[ib])
        {
            c[cSize++] = a[ia];
            ia++;
            ib++;
        }
        else
        {
            if (a[ia] > b[ib])
            {
                ib++;
            }
            else
            {
                ia++;
            }
        }
    }
}

/**
 *  Find rules containing 1 item on the right.
 *  Generate those rules from the items of an input item set
 *
 *  \param minConfidence[in]    minimum confidence
 *  \param L[in]                structure containing "large" item sets
 *  \param itemSetSize[in]      number of items in the input item set
 *  \param items[in]            array of items of the input item set
 *  \param itemsSupport[in]     input item set support
 *  \param leftItems[in]        buffer to store left part of the rule
 *  \param R[in,out]            structure that contains association rules
 *  \param numRulesPtr[in,out]  number of rules
 *  \param numLeftPtr[in,out]   number of items in the left parts of the rules
 *  \param numRightPtr[in,out]  number of items in the right parts of the rules
 *  \param numRulesFound[out]   number of found rules
 *
  */
template <typename algorithmFPType, CpuType cpu>
services::Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::firstPass(double minConfidence, ItemSetList<cpu> * L, size_t itemSetSize,
                                                                                  const size_t * items, size_t itemsSupport, size_t * leftItems,
                                                                                  AssocRule<cpu> * R, size_t & numRules, size_t & numLeft,
                                                                                  size_t & numRight, size_t & numRulesFound)
{
    const ItemSetList<cpu> & L_0    = L[0];
    const ItemSetList<cpu> & L_prev = L[itemSetSize - 1];
    size_t oldNumRules        = numRules;

    for (size_t i = 0; i <= itemSetSize; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            leftItems[j] = items[j];
        }
        for (size_t j = i + 1; j <= itemSetSize; ++j)
        {
            leftItems[j - 1] = items[j];
        }

        const assocrules_itemset<cpu> * left_iset = findItemSet(itemSetSize, leftItems, L_prev);
        DAAL_CHECK(left_iset, services::ErrorNullInput);
        DAAL_CHECK_STATUS_OK(left_iset->ok(), left_iset->getLastStatus());

        const assocrules_itemset<cpu> * right_iset = findItemSet(1, &items[i], L_0);
        DAAL_CHECK(right_iset, services::ErrorNullInput);
        DAAL_CHECK_STATUS_OK(right_iset->ok(), right_iset->getLastStatus());

        double confidence = (double)itemsSupport / (double)(left_iset->support.get());
        if (confidence >= minConfidence)
        {
            R[numRules++].update(left_iset, right_iset, confidence);
            numLeft += itemSetSize;
            numRight++;
        }
    }
    numRulesFound = numRules - oldNumRules;
    return services::Status();
}

/**
 *  Generate rules that have k+1 items on the right from the rules that have k items on the right.
 *
 *  \param minConfidence[in]    minimum confidence
 *  \param L[in]                structure containing "large" item sets
 *  \param right_size[in]       number of items in the right part of the rules (k)
 *  \param itemsSupport[in]     support of the item set superset that contains items of the left
 *                              and right parts of the generated rules
 *  \param leftItems[in]        buffer to store left part of the rule
 *  \param R[in,out]            structure that contains association rules
 *  \param numRulesPtr[in,out]  number of rules
 *  \param numLeftPtr[in,out]   number of items in the left parts of the rules
 *  \param numRightPtr[in,out]  number of items in the right parts of the rules
 *  \param numRulesFound[out]   number of found rules
 *  \param foundPtr[out]        flag: true, if new rules are discovered
 *
  */
template <typename algorithmFPType, CpuType cpu>
services::Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::nextPass(double minConfidence, ItemSetList<cpu> * L, size_t right_size,
                                                                                 size_t itemsSupport, size_t * leftItems, AssocRule<cpu> * R,
                                                                                 size_t & numRules, size_t & numLeft, size_t & numRight,
                                                                                 size_t & numRulesFound, bool & found)
{
    size_t oldNumRules  = numRules;
    size_t n_rules_prev = numRulesFound;
    size_t R_prev_begin = numRules - n_rules_prev;
    found               = false;

    /* Check each pair of rules found on previous step */
    for (size_t i = 0; i < n_rules_prev; ++i)
    {
        for (size_t j = i + 1; j < n_rules_prev; ++j)
        {
            size_t firstIdx       = R_prev_begin + i;
            size_t secondIdx      = R_prev_begin + j;
            size_t * first_items  = R[firstIdx].right->items;
            size_t * second_items = R[secondIdx].right->items;
            /* Check that (right_size-2) items of first rule's right part
               and second rule's right part are equal */
            if (right_size > 2)
            {
                if (assocrules_memcmp<cpu>(first_items, second_items, (right_size - 2)))
                {
                    continue;
                }
                if (first_items[right_size - 2] > second_items[right_size - 2])
                {
                    continue;
                }
            }

            assocrules_itemset<cpu> iset(right_size, first_items, second_items[right_size - 2]);
            DAAL_CHECK_STATUS_OK(iset.ok(), iset.getLastStatus());

            const assocrules_itemset<cpu> * right_iset = findItemSet(right_size, iset.items, L[right_size - 1]);
            first_items                                = R[firstIdx].left->items;
            second_items                               = R[secondIdx].left->items;

            size_t left_size;
            setIntersection(first_items, R[firstIdx].left->size, second_items, R[secondIdx].left->size, leftItems, left_size);
            if (left_size < 1)
            {
                continue;
            }

            const assocrules_itemset<cpu> * left_iset = findItemSet(left_size, leftItems, L[left_size - 1]);

            double confidence = (double)itemsSupport / (double)(left_iset->support.get());
            if (confidence >= minConfidence)
            {
                found = true;
                R[numRules++].update(left_iset, right_iset, confidence);
                numLeft += left_size;
                numRight += right_size;
            }
        }
    }
    numRulesFound = numRules - oldNumRules;
    return services::Status();
}

/**
 *  Generate association rules from "large" item sets
 *
 *  \param minConfidence[in]    minimum confidence
 *  \param L_size[in]           length of the array L
 *  \param minItemsetSize[in]   minimal number of items in the "large" itemsets
 *  \param L[in]                structure that contains "large" itemsets
 *  \param R[out]               structure that contains association rules
 *  \param numRules[out]        number of discovered association rules
 *  \param numLeft[out]         total number of items in the left parts of association rules
 *  \param numRight[out]        total number of items in the right parts of association rules
 *
 */
template <typename algorithmFPType, CpuType cpu>
services::Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::generateRules(double minConfidence, size_t minItemsetSize, size_t L_size,
                                                                                      ItemSetList<cpu> * L, AssocRule<cpu> * R, size_t & numRules,
                                                                                      size_t & numLeft, size_t & numRight)
{
    numRules = 0;
    numLeft  = 0;
    numRight = 0;

    TArray<size_t, cpu> leftItemsAr(L_size);
    size_t * leftItems = leftItemsAr.get();
    if (!leftItems) return services::ErrorMemoryAllocationFailed;

    /* Generate all association rules */
    size_t startItemsetSize = 1;
    if (minItemsetSize > startItemsetSize)
    {
        startItemsetSize = minItemsetSize - 1;
    }
    for (size_t iset_size = startItemsetSize; iset_size < L_size; ++iset_size)
    {
        ItemSetList<cpu> & L_cur = L[iset_size];

        for (const auto * current = L_cur.start; current != nullptr; current = current->next())
        {
            const size_t * items = current->itemSet()->items;
            size_t itemsSupport  = current->itemSet()->support.get();
            size_t n_rules_prev  = 0;

            /* Find rules that have 1 item in the right part */
            services::Status statFirstPass =
                firstPass(minConfidence, L, iset_size, items, itemsSupport, leftItems, R, numRules, numLeft, numRight, n_rules_prev);
            DAAL_CHECK_STATUS_OK(statFirstPass.ok(), statFirstPass);

            bool found = (n_rules_prev > 0);
            for (size_t right_size = 2; right_size <= iset_size && found; ++right_size)
            {
                /* Find rules that have right_size items in the right part */
                services::Status statNextPass =
                    nextPass(minConfidence, L, right_size, itemsSupport, leftItems, R, numRules, numLeft, numRight, n_rules_prev, found);

                DAAL_CHECK_STATUS_OK(statNextPass.ok(), statNextPass);
            }
        }
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
void AssociationRulesKernel<apriori, algorithmFPType, cpu>::setRules(AssocRule<cpu> ** rulesArray, size_t numRules, int * rleft, int * rright,
                                                                     algorithmFPType * rconf)
{
    size_t i_rule = 0, i_left = 0, i_right = 0;
    for (i_rule = 0; i_rule < numRules; ++i_rule)
    {
        size_t * leftItems  = rulesArray[i_rule]->left->items;
        size_t left_size    = rulesArray[i_rule]->left->size;
        size_t * rightItems = rulesArray[i_rule]->right->items;
        size_t right_size   = rulesArray[i_rule]->right->size;
        for (size_t iitem = 0; iitem < left_size; ++iitem, ++i_left)
        {
            rleft[2 * i_left]     = i_rule;
            rleft[2 * i_left + 1] = leftItems[iitem];
        }
        for (size_t iitem = 0; iitem < right_size; ++iitem, ++i_right)
        {
            rright[2 * i_right]     = i_rule;
            rright[2 * i_right + 1] = rightItems[iitem];
        }
        rconf[i_rule] = (algorithmFPType)rulesArray[i_rule]->confidence;
    }
}

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
