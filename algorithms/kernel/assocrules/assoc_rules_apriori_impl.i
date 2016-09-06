/* file: assoc_rules_apriori_impl.i */
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
//  Implementation of auxiliary functions for association rules
//  Apriori method.
//--
*/

#ifndef __ASSOC_RULES_APRIORI_IMPL_I__
#define __ASSOC_RULES_APRIORI_IMPL_I__

#include "service_micro_table.h"
#include "service_sort.h"

#include "assoc_rules_apriori_mine_impl.i"
#include "assoc_rules_apriori_discover_impl.i"

using namespace daal::algorithms::internal;

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{

template <typename interm, CpuType cpu>
void AssociationRulesKernel<apriori, interm, cpu>::compute(const NumericTable *a, size_t nr, NumericTable *r[],
                                                           const daal::algorithms::Parameter *algParameter)
{
    NumericTable *dataTable = const_cast<NumericTable *>(a);
    const daal::algorithms::association_rules::Parameter *parameter =
            static_cast<const daal::algorithms::association_rules::Parameter *>(algParameter);
    double minSupport = parameter->minSupport;
    size_t minItemsetSize = (parameter->minItemsetSize ? parameter->minItemsetSize : 1);

    NumericTable *largeItemsetsTable        = r[0];
    NumericTable *largeItemsetsSupportTable = r[1];

    /* Create association rules data set from input numeric table */
    assocrules_dataset<cpu> data(dataTable, parameter->nTransactions, parameter->nUniqueItems, minSupport);

    ItemSetList<cpu> *L = new ItemSetList<cpu>[data.numOfUniqueItems];
    AssocRule<cpu> *R = NULL;

    /* Find "large" itemsets */
    size_t L_size = 0;
    size_t maxItemsetSize    = ((parameter->maxItemsetSize == 0) ? (size_t) - 1 : parameter->maxItemsetSize);
    findLargeItemsets((size_t)daal::internal::Math<double,cpu>::sCeil(minSupport * data.numOfTransactions), maxItemsetSize, data, L, &L_size);
    if (this->_errors->size() > 0) { return; }

    /* Allocate memory to store "large" itemsets */
    size_t nLargeItemSets = 0;
    size_t nItemInLargeItemSets = 0;
    allocateItemsetsTableData(L, L_size, minItemsetSize, largeItemsetsTable, largeItemsetsSupportTable,
                              &nLargeItemSets, &nItemInLargeItemSets);
    if (this->_errors->size() > 0) { return; }
    if (L_size == 0) { return; }

    /* Write "large" itemsets into resulting tables */
    writeItemsetsTableData(L, L_size, minItemsetSize, parameter->itemsetsOrder,
                           largeItemsetsTable, largeItemsetsSupportTable);
    if (this->_errors->size() > 0) { return; }

    if (parameter->discoverRules)
    {
        size_t maxRulesNum = L[0].size * (L[0].size - 1);
        for (size_t i = 1; i < L_size; i++)
        {
            size_t exp2LSize = ((size_t)1 << (i + 1)) - (size_t)2;
            if (exp2LSize < 2) { exp2LSize = 2; }
            maxRulesNum += L[i].size * (exp2LSize - 1) * (exp2LSize);
        }

        R = new AssocRule<cpu>[maxRulesNum];
        if (!R) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }
        size_t nRules = 0;            /*<! Number of association rules */
        size_t nLeft  = 0;            /*<! Number of items in left parts of the rules */
        size_t nRight = 0;            /*<! Number of items in right parts of the rules */
        double minConfidence = parameter->minConfidence;
        generateRules(minConfidence, minItemsetSize, L_size, &L, R, &nRules, &nLeft, &nRight);
        if(this->_errors->size() != 0) { return; }

        NumericTable *leftItemsTable    = r[2];
        NumericTable *rightItemsTable   = r[3];
        NumericTable *confidenceTable   = r[4];

        /* Allocate memory to store association rules */
        allocateRulesTableData(leftItemsTable, rightItemsTable, confidenceTable, nLeft, nRight, nRules);
        if (this->_errors->size() > 0) { return; }
        if (nRules == 0) { return; }

        /* Write association rules into resulting tables */
        writeRulesTableData(R, parameter->rulesOrder, leftItemsTable, rightItemsTable, confidenceTable);
        if (this->_errors->size() > 0) { return; }

        delete[] R;
    }

    for (size_t i = 0; i < L_size; i++)
    {
        L[i].remove();
    }
    delete[] L;
    return;
}

template <typename interm, CpuType cpu>
void AssociationRulesKernel<apriori, interm, cpu>::findLargeItemsets(size_t minSupport, size_t maxItemsetSize,
                                                                     assocrules_dataset<cpu> &data,
                                                                     ItemSetList<cpu> *L, size_t *L_size_ptr)
{
    size_t L_size = *L_size_ptr;
    /* Form list of "large" item sets of size 1 from the unique items
       which count is not less than minimum count 'imin_s' */
    firstPass(minSupport, data, L, &L_size);
    //if(this->_errors->size() != 0) { return; }

    if (L_size == 0) { return; }

    /* Find "large" item sets of size k+1 from itemsets of size k */
    size_t k = 1;
    bool found = false;
    hash_tree<cpu> *C_tree = NULL;
    do
    {
        C_tree = nextPass(minSupport, k++, data, L, &L_size, &found, C_tree);
        //if(this->_errors->size() != 0) { return; }
    }
    while (found && k < maxItemsetSize);

    if (C_tree) delete C_tree;
    *L_size_ptr = L_size;
}

template <typename interm, CpuType cpu>
void AssociationRulesKernel<apriori, interm, cpu>::allocateItemsetsTableData(ItemSetList<cpu> *L, size_t L_size,
         size_t minItemsetSize, NumericTable *largeItemsetsTable, NumericTable *largeItemsetsSupportTable,
         size_t *nLargeItemSetsPtr, size_t *nItemInLargeItemSetsPtr)
{
    if (L_size == 0)
    {
        largeItemsetsTable->setNumberOfRows(0);
        largeItemsetsSupportTable->setNumberOfRows(0);
        return;
    }

    size_t nLargeItemSets = *nLargeItemSetsPtr;
    size_t nItemInLargeItemSets = *nItemInLargeItemSetsPtr;

    for (size_t i = minItemsetSize - 1; i < L_size; i++)
    {
        size_t L_number = L[i].size;
        nLargeItemSets += L_number;
        nItemInLargeItemSets += (i + 1) * L_number;
    }

    allocateTableData(largeItemsetsTable, nItemInLargeItemSets, services::ErrorAprioriIncorrectItemsetTableSize);
    if(this->_errors->size() > 0) { return; }

    allocateTableData(largeItemsetsSupportTable, nLargeItemSets, services::ErrorAprioriIncorrectSupportTableSize);
    if(this->_errors->size() > 0) { return; }

    *nLargeItemSetsPtr = nLargeItemSets;
    *nItemInLargeItemSetsPtr = nItemInLargeItemSets;
}

template <typename interm, CpuType cpu>
void AssociationRulesKernel<apriori, interm, cpu>::writeItemsetsTableData(ItemSetList<cpu> *L, size_t L_size,
        size_t minItemsetSize, ItemsetsOrder itemsetsOrder,
        NumericTable *largeItemsetsTable, NumericTable *largeItemsetsSupportTable)
{
    BlockMicroTable<int, writeOnly, cpu> mtLargeItemsets(largeItemsetsTable);
    BlockMicroTable<int, writeOnly, cpu> mtLargeItemsetsSupport(largeItemsetsSupportTable);

    int *largeItemsetsData;
    int *largeItemsetsSupportData;

    size_t nLargeItemSets = largeItemsetsSupportTable->getNumberOfRows();

    mtLargeItemsets.getBlockOfRows(0, largeItemsetsTable->getNumberOfRows(), &largeItemsetsData);
    mtLargeItemsetsSupport.getBlockOfRows(0, nLargeItemSets, &largeItemsetsSupportData);

    /* Array of pointers to "large" itemsets */
    assocrules_itemset<cpu> ** itemsetsArray = (assocrules_itemset<cpu> **)daal::services::daal_malloc(
        nLargeItemSets * sizeof(assocrules_itemset<cpu> *));
    if (!itemsetsArray) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    size_t iset_idx = 0;
    for (size_t i = minItemsetSize - 1, k = 0; i < L_size; i++)
    {
        for (auto *current = L[i].start; current != NULL; current = current->next, iset_idx++, k++)
        {
            itemsetsArray[k] = current->itemSet;
        }
    }

    if (itemsetsOrder == itemsetsSortedBySupport)
    {
        qSort<assocrules_itemset<cpu> *, cpu>(nLargeItemSets, itemsetsArray, compareItemsetsBySupport<cpu>);
    }

    size_t item_idx = 0;
    for (size_t i = 0; i < nLargeItemSets; i++)
    {
        assocrules_itemset<cpu> *itemSet = itemsetsArray[i];
        size_t *items = itemSet->items;
        for (size_t j = 0; j < itemSet->size; j++, item_idx++)
        {
            largeItemsetsData[2 * item_idx]     = i;
            largeItemsetsData[2 * item_idx + 1] = items[j];
        }
        largeItemsetsSupportData[2 * i]     = i;
        largeItemsetsSupportData[2 * i + 1] = (int)itemSet->support.get();
    }

    mtLargeItemsets.release();
    mtLargeItemsetsSupport.release();

    daal::services::daal_free(itemsetsArray);
}

template <typename interm, CpuType cpu>
void AssociationRulesKernel<apriori, interm, cpu>::allocateRulesTableData(
        NumericTable *leftItemsTable, NumericTable *rightItemsTable, NumericTable *confidenceTable,
        size_t nLeft, size_t nRight, size_t nRules)
{
    if (nRules == 0)
    {
        leftItemsTable->setNumberOfRows(0);
        rightItemsTable->setNumberOfRows(0);
        confidenceTable->setNumberOfRows(0);
        return;
    }

    allocateTableData(leftItemsTable, nLeft, services::ErrorAprioriIncorrectLeftRuleTableSize);
    if(this->_errors->size() > 0) { return; }

    allocateTableData(rightItemsTable, nRight, services::ErrorAprioriIncorrectRightRuleTableSize);
    if(this->_errors->size() > 0) { return; }

    allocateTableData(confidenceTable, nRules, services::ErrorAprioriIncorrectConfidenceTableSize);
    if(this->_errors->size() > 0) { return; }
}

template <typename interm, CpuType cpu>
void AssociationRulesKernel<apriori, interm, cpu>::writeRulesTableData(AssocRule<cpu> *R, RulesOrder rulesOrder,
        NumericTable *leftItemsTable, NumericTable *rightItemsTable, NumericTable *confidenceTable)
{
    BlockMicroTable<int, writeOnly, cpu> mtLeftItems(leftItemsTable);
    BlockMicroTable<int, writeOnly, cpu> mtRightItems(rightItemsTable);
    FeatureMicroTable<interm, writeOnly, cpu> mtConfidence(confidenceTable);

    int *leftItems, *rightItems;
    interm *confidence;

    size_t nRules = confidenceTable->getNumberOfRows();
    mtLeftItems.getBlockOfRows(0, leftItemsTable->getNumberOfRows(), &leftItems);
    mtRightItems.getBlockOfRows(0, rightItemsTable->getNumberOfRows(), &rightItems);
    mtConfidence.getBlockOfColumnValues(0, 0, nRules, &confidence);

    AssocRule<cpu> **rulesArray = (AssocRule<cpu> **)daal::services::daal_malloc(nRules * sizeof(AssocRule<cpu> *));
    if (!rulesArray) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    for (size_t i = 0; i < nRules; i++)
    {
        rulesArray[i] = R + i;
    }

    if (rulesOrder == rulesSortedByConfidence)
    {
        qSort<AssocRule<cpu> *, cpu>(nRules, rulesArray, compareRulesByConfidence<cpu>);
    }
    setRules(rulesArray, nRules, leftItems, rightItems, confidence);

    mtLeftItems.release();
    mtRightItems.release();
    mtConfidence.release();
    daal::services::daal_free(rulesArray);
}

template <typename interm, CpuType cpu>
void AssociationRulesKernel<apriori, interm, cpu>::allocateTableData(NumericTable *table, size_t nRows, const services::ErrorID &errorcode)
{
    if (table->getDataMemoryStatus() == NumericTableIface::notAllocated)
    {
        table->setNumberOfRows(nRows);
        table->allocateDataMemory();
        if(this->_errors->size() > 0) { return; }
    }
    else
    {
        if (table->getNumberOfRows() < nRows)
        {
            this->_errors->add(errorcode);
            return;
        }
        table->setNumberOfRows(nRows);
    }
    return;
}

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
