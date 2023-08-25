/* file: assoc_rules_apriori_impl.i */
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
//  Implementation of auxiliary functions for association rules
//  Apriori method.
//--
*/

#ifndef __ASSOC_RULES_APRIORI_IMPL_I__
#define __ASSOC_RULES_APRIORI_IMPL_I__

#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_sort.h"

#include "src/algorithms/assocrules/assoc_rules_apriori_mine_impl.i"
#include "src/algorithms/assocrules/assoc_rules_apriori_discover_impl.i"

#include "src/services/service_data_utils.h"

using namespace daal::algorithms::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::compute(const NumericTable * a, NumericTable * r[],
                                                                      const daal::algorithms::Parameter * algParameter)
{
    NumericTable * dataTable = const_cast<NumericTable *>(a);
    const daal::algorithms::association_rules::Parameter * parameter =
        static_cast<const daal::algorithms::association_rules::Parameter *>(algParameter);
    const double minSupport = parameter->minSupport;
    size_t minItemsetSize   = (parameter->minItemsetSize ? parameter->minItemsetSize : 1);

    NumericTable * largeItemsetsTable        = r[0];
    NumericTable * largeItemsetsSupportTable = r[1];

    /* Create association rules data set from input numeric table */
    assocrules_dataset<cpu> data(dataTable, parameter->nTransactions, parameter->nUniqueItems, minSupport);
    DAAL_CHECK_STATUS_OK(data.ok(), data.getLastStatus());

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, data.numOfUniqueItems, sizeof(ItemSetList<cpu>));

    TArray<ItemSetList<cpu>, cpu> L(data.numOfUniqueItems);
    DAAL_CHECK(L.get(), ErrorMemoryAllocationFailed);
    for (size_t i = 0, n = L.size(); i < n; ++i) L[i].setDataOwner(true);

    /* Find "large" itemsets */
    size_t L_size         = 0;
    size_t maxItemsetSize = ((parameter->maxItemsetSize == 0) ? (size_t)-1 : parameter->maxItemsetSize);
    double ceil           = daal::internal::MathInst<double, cpu>::sCeil(minSupport * data.numOfTransactions);
    DAAL_ASSERT(ceil >= 0)
    services::Status statLargeItemset = findLargeItemsets((size_t)ceil, maxItemsetSize, data, L.get(), L_size);
    DAAL_CHECK_STATUS_OK(statLargeItemset.ok(), statLargeItemset);
    DAAL_ASSERT(L_size > 0);

    /* Allocate memory to store "large" itemsets */
    size_t nLargeItemSets       = 0;
    size_t nItemInLargeItemSets = 0;
    Status s;
    DAAL_CHECK_STATUS(s, allocateItemsetsTableData(L.get(), L_size, minItemsetSize, largeItemsetsTable, largeItemsetsSupportTable, nLargeItemSets,
                                                   nItemInLargeItemSets));

    /* Write "large" itemsets into resulting tables */
    DAAL_CHECK_STATUS(
        s, writeItemsetsTableData(L.get(), L_size, minItemsetSize, parameter->itemsetsOrder, *largeItemsetsTable, *largeItemsetsSupportTable));

    if (parameter->discoverRules)
    {
        size_t maxRulesNum = L[0].size * (L[0].size - 1);
        for (size_t i = 1; i < L_size; i++)
        {
            size_t exp2LSize = ((size_t)1 << (i + 1)) - (size_t)2;
            if (exp2LSize < 2)
            {
                exp2LSize = 2;
            }
            maxRulesNum += L[i].size * (exp2LSize - 1) * (exp2LSize);
        }

        TArray<AssocRule<cpu>, cpu> R(maxRulesNum);
        DAAL_CHECK(R.get(), ErrorMemoryAllocationFailed);

        size_t nRules                 = 0; /*<! Number of association rules */
        size_t nLeft                  = 0; /*<! Number of items in left parts of the rules */
        size_t nRight                 = 0; /*<! Number of items in right parts of the rules */
        double minConfidence          = parameter->minConfidence;
        services::Status statGenRules = generateRules(minConfidence, minItemsetSize, L_size, L.get(), R.get(), nRules, nLeft, nRight);
        DAAL_CHECK_STATUS_OK(statGenRules.ok() && !!nRules, statGenRules);

        NumericTable * leftItemsTable  = r[2];
        NumericTable * rightItemsTable = r[3];
        NumericTable * confidenceTable = r[4];

        /* Allocate memory to store association rules */
        DAAL_CHECK_STATUS(s, allocateRulesTableData(leftItemsTable, rightItemsTable, confidenceTable, nLeft, nRight, nRules));

        /* Write association rules into resulting tables */
        DAAL_CHECK_STATUS(s, writeRulesTableData(R.get(), parameter->rulesOrder, leftItemsTable, rightItemsTable, confidenceTable));
    }
    return s;
}

template <typename algorithmFPType, CpuType cpu>
services::Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::findLargeItemsets(size_t minSupport, size_t maxItemsetSize,
                                                                                          assocrules_dataset<cpu> & data, ItemSetList<cpu> * L,
                                                                                          size_t & L_size)
{
    /* Form list of "large" item sets of size 1 from the unique items
       which count is not less than minimum count 'imin_s' */
    services::Status statFirstPass = firstPass(minSupport, data, *L);
    DAAL_CHECK_STATUS_OK(statFirstPass.ok(), statFirstPass);

    L_size = 1;
    /* Find "large" item sets of size k+1 from itemsets of size k */
    size_t k                = 1;
    bool bFound             = false;
    hash_tree<cpu> * C_tree = NULL;
    do
    {
        services::Status s;
        C_tree = nextPass(minSupport, k++, data, L, L_size, bFound, C_tree, s);
        DAAL_CHECK_STATUS_OK(s.ok(), s);
    } while (bFound && k < maxItemsetSize);

    delete C_tree;
    return (L_size > 0) ? services::Status() : services::ErrorAprioriIncorrectInputData;
}

template <typename algorithmFPType, CpuType cpu>
Status allocateTableData(NumericTable * table, size_t nRows, const services::ErrorID errorcode)
{
    if (table->getDataMemoryStatus() == NumericTableIface::notAllocated) return table->resize(nRows);
    if (table->getNumberOfRows() < nRows) return Status(errorcode);
    return table->resize(nRows);
}

template <typename algorithmFPType, CpuType cpu>
Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::allocateItemsetsTableData(ItemSetList<cpu> * L, size_t L_size, size_t minItemsetSize,
                                                                                        NumericTable * largeItemsetsTable,
                                                                                        NumericTable * largeItemsetsSupportTable,
                                                                                        size_t & nLargeItemSets, size_t & nItemInLargeItemSets)
{
    Status s;
    if (L_size == 0)
    {
        largeItemsetsTable->resize(0);
        largeItemsetsSupportTable->resize(0);
        return s;
    }

    for (size_t i = minItemsetSize - 1; i < L_size; i++)
    {
        size_t L_number = L[i].size;
        nLargeItemSets += L_number;
        nItemInLargeItemSets += (i + 1) * L_number;
    }

    DAAL_CHECK_STATUS(s, (allocateTableData<algorithmFPType, cpu>(largeItemsetsTable, nItemInLargeItemSets, ErrorAprioriIncorrectItemsetTableSize)));
    return allocateTableData<algorithmFPType, cpu>(largeItemsetsSupportTable, nLargeItemSets, ErrorAprioriIncorrectSupportTableSize);
}

template <typename algorithmFPType, CpuType cpu>
Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::writeItemsetsTableData(ItemSetList<cpu> * L, size_t L_size, size_t minItemsetSize,
                                                                                     ItemsetsOrder itemsetsOrder, NumericTable & largeItemsetsTable,
                                                                                     NumericTable & largeItemsetsSupportTable)
{
    const size_t nLargeItemSets = largeItemsetsSupportTable.getNumberOfRows();

    WriteOnlyRows<int, cpu> mtLargeItemsets(largeItemsetsTable, 0, largeItemsetsTable.getNumberOfRows());
    DAAL_CHECK_BLOCK_STATUS(mtLargeItemsets);
    WriteOnlyRows<int, cpu> mtLargeItemsetsSupport(largeItemsetsSupportTable, 0, nLargeItemSets);
    DAAL_CHECK_BLOCK_STATUS(mtLargeItemsetsSupport);

    int * largeItemsetsData        = mtLargeItemsets.get();
    int * largeItemsetsSupportData = mtLargeItemsetsSupport.get();

    /* Array of pointers to "large" itemsets */
    typedef const assocrules_itemset<cpu> * ItemsetConstPtr;
    TArray<ItemsetConstPtr, cpu> itemsetsArray(nLargeItemSets);
    DAAL_CHECK(itemsetsArray.get(), ErrorMemoryAllocationFailed);

    size_t iset_idx = 0;
    for (size_t i = minItemsetSize - 1, k = 0; i < L_size; i++)
    {
        for (auto * current = L[i].start; current != NULL; current = current->next(), iset_idx++, k++)
        {
            itemsetsArray[k] = current->itemSet();
        }
    }

    if (itemsetsOrder == itemsetsSortedBySupport)
    {
        qSort<ItemsetConstPtr, cpu>(nLargeItemSets, itemsetsArray.get(), compareItemsetsBySupport<cpu>);
    }

    size_t item_idx = 0;
    for (size_t i = 0; i < nLargeItemSets; i++)
    {
        ItemsetConstPtr itemSet = itemsetsArray[i];
        for (size_t j = 0; j < itemSet->size; j++, item_idx++)
        {
            largeItemsetsData[2 * item_idx]     = i;
            largeItemsetsData[2 * item_idx + 1] = itemSet->items[j];
        }
        largeItemsetsSupportData[2 * i] = i;
        DAAL_ASSERT(itemSet->support.get() <= services::internal::MaxVal<int>::get())
        largeItemsetsSupportData[2 * i + 1] = (int)itemSet->support.get();
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::allocateRulesTableData(NumericTable * leftItemsTable, NumericTable * rightItemsTable,
                                                                                     NumericTable * confidenceTable, size_t nLeft, size_t nRight,
                                                                                     size_t nRules)
{
    Status s;
    if (nRules == 0)
    {
        leftItemsTable->resize(0);
        rightItemsTable->resize(0);
        confidenceTable->resize(0);
        return s;
    }

    DAAL_CHECK_STATUS(s, (allocateTableData<algorithmFPType, cpu>(leftItemsTable, nLeft, ErrorAprioriIncorrectLeftRuleTableSize)));
    DAAL_CHECK_STATUS(s, (allocateTableData<algorithmFPType, cpu>(rightItemsTable, nRight, ErrorAprioriIncorrectRightRuleTableSize)));
    return allocateTableData<algorithmFPType, cpu>(confidenceTable, nRules, ErrorAprioriIncorrectConfidenceTableSize);
}

template <typename algorithmFPType, CpuType cpu>
Status AssociationRulesKernel<apriori, algorithmFPType, cpu>::writeRulesTableData(AssocRule<cpu> * R, RulesOrder rulesOrder,
                                                                                  NumericTable * leftItemsTable, NumericTable * rightItemsTable,
                                                                                  NumericTable * confidenceTable)
{
    WriteOnlyRows<int, cpu> mtLeftItems(leftItemsTable, 0, leftItemsTable->getNumberOfRows());
    DAAL_CHECK_BLOCK_STATUS(mtLeftItems);
    WriteOnlyRows<int, cpu> mtRightItems(rightItemsTable, 0, rightItemsTable->getNumberOfRows());
    DAAL_CHECK_BLOCK_STATUS(mtRightItems);
    int * leftItems     = mtLeftItems.get();
    int * rightItems    = mtRightItems.get();
    const size_t nRules = confidenceTable->getNumberOfRows();

    TArray<AssocRule<cpu> *, cpu> rulesArray(nRules);
    DAAL_CHECK(rulesArray.get(), ErrorMemoryAllocationFailed);
    for (size_t i = 0; i < nRules; i++)
    {
        rulesArray[i] = R + i;
    }

    if (rulesOrder == rulesSortedByConfidence)
    {
        qSort<AssocRule<cpu> *, cpu>(nRules, rulesArray.get(), compareRulesByConfidence<cpu>);
    }
    WriteOnlyColumns<algorithmFPType, cpu> mtConfidence(confidenceTable, 0, 0, nRules);
    DAAL_CHECK_BLOCK_STATUS(mtConfidence);
    algorithmFPType * confidence = mtConfidence.get();
    setRules(rulesArray.get(), nRules, leftItems, rightItems, confidence);
    return Status();
}

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
