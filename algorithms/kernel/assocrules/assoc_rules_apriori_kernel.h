/* file: assoc_rules_apriori_kernel.h */
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
//  Declaration of template function that computes association rules results.
//--
*/

#ifndef __ASSOC_RULES_APRIORI_KERNEL_H__
#define __ASSOC_RULES_APRIORI_KERNEL_H__

#include "assoc_rules_kernel.h"

#include "assoc_rules_apriori_itemset.i"
#include "assoc_rules_apriori_types.i"
#include "assoc_rules_apriori_tree.i"

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{
/**
 *  Structure that contains kernels for Apriori association rules mining
 */
template <typename interm, CpuType cpu>
class AssociationRulesKernel<apriori, interm, cpu> : public Kernel
{
public:
    /** Find "large" item sets and build association rules */
    void compute(const NumericTable *a, size_t nr, NumericTable *r[], const daal::algorithms::Parameter *parameter);
protected:
    void findLargeItemsets(size_t minSupport, size_t maxItemsetSize, assocrules_dataset<cpu> &data, ItemSetList<cpu> *L, size_t *L_size);

    void allocateItemsetsTableData(ItemSetList<cpu> *L, size_t L_size, size_t minItemsetSize,
                                   NumericTable *largeItemsetsTable, NumericTable *largeItemsetsSupportTable,
                                   size_t *nLargeItemSets, size_t *nItemInLargeItemSets);

    void writeItemsetsTableData(ItemSetList<cpu> *L, size_t L_size, size_t minItemsetSize, ItemsetsOrder itemsetsOrder,
                                NumericTable *largeItemsetsTable, NumericTable *largeItemsetsSupportTable);

    void allocateRulesTableData(NumericTable *leftItemsTable, NumericTable *rightItemsTable, NumericTable *confidenceTable,
                                size_t nLeft, size_t nRight, size_t nRules);

    void writeRulesTableData(AssocRule<cpu> *R, RulesOrder rulesOrder, NumericTable *leftItemsTable, NumericTable *rightItemsTable,
                             NumericTable *confidenceTable);

    void allocateTableData(NumericTable *table, size_t nRows, const services::ErrorID &errorcode);

    /*
     *  Auxiliary methods for large item sets mining
     */

    /** Find "large" itemsets of size 1 */
    void firstPass(size_t imin_s, assocrules_dataset<cpu> &data, ItemSetList<cpu> *L, size_t *L_size);

    /** Generate "large" item sets of size k+1 from "large" item sets of size k */
    hash_tree<cpu> * nextPass(size_t imin_s, size_t iset_size, assocrules_dataset<cpu> &data, ItemSetList<cpu> *L, size_t *L_size,
                              bool *found_ptr, hash_tree<cpu> *C_tree);

    /** Test that all {n-1}-item subsets of {n}-item set are "large" item sets */
    bool pruneCandidate(size_t iset_size, const size_t *cadidate, size_t *subset, hash_tree<cpu> &C_tree);

    size_t binarySearch(size_t nUniqueItems, assocRulesUniqueItem<cpu> *uniqueItems, size_t itemID);

    assocrules_itemset<cpu> * genCandidate(size_t iset_size, size_t *first_items, size_t second_item,
                                           size_t *subset_buf, hash_tree<cpu> *C_tree);

    /** Generate candidate itemsets of size iset_size+1 from "large" itemsets of size iset_size */
    void genCandidates(size_t iset_size, ItemSetList<cpu> *L, bool *found_ptr, hash_tree<cpu> *C_tree,
                       size_t nUniqueItems, assocRulesUniqueItem<cpu> *uniqueItems);

    /** Generate all subsets of size iset_size from a transaction and hash those subsets
        using hash tree of candidate itemsets C_tree to increment support values of candidates */
    void genSubset(size_t transactionSize, const size_t *items, size_t iset_size,
                   size_t *subset, size_t *idx, hash_tree<cpu> &C_tree, size_t &large_count);

    /** Remove candidate itemsets which support is less then minimum support
        from array of "large" item sets */
    void prune(size_t imin_s, size_t iset_size, assocrules_dataset<cpu> &data, ItemSetList<cpu> *L, hash_tree<cpu> *C_tree);

    /*
     *  Auxiliary methods for association rules discovery
     */

    /** Find item set in the list of item sets */
    assocrules_itemset<cpu> *findItemSet(size_t items_size, size_t *items, ItemSetList<cpu> &L_cur);

    /** Find intersection between two item sets */
    void setIntersection(size_t *a, size_t aSize, size_t *b, size_t bSize, size_t *c, size_t *cSizePtr);

    /** Find rules containing 1 item on the right */
    void firstPass(double minConfidence, ItemSetList<cpu> **L, size_t itemSetSize, size_t *items, size_t itemsSupport,
                   size_t *leftItems, AssocRule<cpu> *R, size_t *numRulesPtr,
                   size_t *numLeftPtr, size_t *numRightPtr, size_t *numRulesFound);

    /** Generate rules that have k+1 items on the right from the rules that have k items on the right */
    void nextPass(double minConfidence, ItemSetList<cpu> **L, size_t right_size,
                  size_t itemsSupport, size_t *leftItems, AssocRule<cpu> *R,
                  size_t *numRulesPtr, size_t *numLeftPtr, size_t *numRightPtr, size_t *numRulesFound,
                  bool *foundPtr);

    /** Generate association rules from "large" item sets */
    void generateRules(double minConfidence, size_t minItemsetSize, size_t L_size, ItemSetList<cpu> **L,
                       AssocRule<cpu> *R, size_t *numRulesPtr, size_t *numLeftPtr, size_t *numRightPtr);

    /** Store association rules into continuous memory */
    void setRules(AssocRule<cpu> **R, size_t numRules, int *rleft, int *rright, interm *rconf);
};

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
