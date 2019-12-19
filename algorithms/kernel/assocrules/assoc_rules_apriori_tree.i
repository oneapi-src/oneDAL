/* file: assoc_rules_apriori_tree.i */
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
//  Declarations of hash tree structure that is used in Apriori algorithm
//--
*/

#ifndef __ASSOC_RULES_APRIORI_TREE_I__
#define __ASSOC_RULES_APRIORI_TREE_I__

#include "threading.h"
#include "service_memory.h"
#include "assoc_rules_apriori_itemset.i"
#include "assoc_rules_apriori_types.i"

namespace daal
{
namespace algorithms
{
namespace association_rules
{
namespace internal
{
template <CpuType cpu>
int iLog2(int x)
{
    int res = 1;
    while ((x >>= 1) != 0)
    {
        res++;
    }
    return res;
}

/**
 *  \brief Hash tree internal node
 */
template <CpuType cpu>
struct hash_tree_node
{
    DAAL_NEW_DELETE();
    hash_tree_node() { mask = 0; }
    __int64 mask; /*<! Bit mask that specifies children nodes */
};

/**
 *  \brief Hash tree leaf node
 */
template <CpuType cpu>
struct hash_tree_leaf
{
    DAAL_NEW_DELETE();
    hash_tree_leaf() : iset_list() {}

    /*<! List of itemsets that were hashed to this node */
    ItemSetList<cpu> iset_list;
};

/**
 *  \brief Hash tree
 */
template <CpuType cpu>
struct hash_tree
{
    DAAL_NEW_DELETE();
    /** \brief Construct hash tree from a list of "candidate" itemsets */
    hash_tree(int h, ItemSetList<cpu> & L_cur) : height(h > maxHeight ? maxHeight : h)
    {
        logOrder = iLog2<cpu>(L_cur.size) / height;
        if (logOrder > maxLogOrder)
        {
            logOrder = maxLogOrder;
        }
        if (logOrder < minLogOrder)
        {
            logOrder = minLogOrder;
        }
        order     = (1 << logOrder);
        order_m1  = order - 1;
        n_nodes_i = (int *)daal::services::daal_malloc(height * sizeof(int));

        if (n_nodes_i)
        {
            n_nodes_i[0] = 1;
            n_nodes      = 1;
            for (int i = 1; i < height; i++)
            {
                n_nodes_i[i] = n_nodes_i[i - 1] * order;
                n_nodes += n_nodes_i[i];
            }
            n_leaves = n_nodes_i[height - 1] * order;

            root   = new hash_tree_node<cpu>[n_nodes];
            leaves = new hash_tree_leaf<cpu>[n_leaves];

            if (root && leaves)
            {
                for (auto * current = L_cur.start; current; current = current->next())
                {
                    insert_itemset(current->itemSet());
                }

                _status = services::Status();
            }
            else
            {
                if (root) delete[] root;

                if (leaves) delete[] leaves;

                root    = NULL;
                leaves  = NULL;
                _status = services::ErrorMemoryAllocationFailed;
            }
        }
        else
        {
            _status = services::ErrorMemoryAllocationFailed;
        }
    }

    ~hash_tree()
    {
        daal::services::daal_free(n_nodes_i);
        delete[] root;
        delete[] leaves;
        n_nodes_i = nullptr;
        root      = nullptr;
        leaves    = nullptr;
    }

    /** \brief Hash function that is applied to each item in itemset */
    size_t hash_func(size_t x) const { return ((x ^ (x >> 5)) & order_m1); }

    /** \brief Check if the node has child corresponding to hashing result */
    bool is_child(size_t node_idx, size_t hash_res) const { return ((1 << hash_res) & root[node_idx].mask); }

    /** \brief Update node's cildren mask using hashing result */
    void update_node_mask(size_t node_idx, size_t hash_res)
    {
        if (!is_child(node_idx, hash_res))
        {
            /* Update children mask of the root node if needed */
            root[node_idx].mask |= (1 << hash_res);
        }
    }

    /** \brief Insert itemset into the tree */
    void insert_itemset(assocrules_itemset<cpu> * iset)
    {
        size_t hash_res;
        size_t * items = iset->items;
        size_t base = 0, offset = 0;
        hash_res = hash_func(items[0]);
        update_node_mask(0, hash_res);
        offset = hash_res;
        base   = 1;
        for (int i = 1; i < height - 1; i++)
        {
            hash_res = hash_func(items[i]);
            update_node_mask(base + offset, hash_res);
            base += n_nodes_i[i];
            offset = order * offset + hash_res;
        }
        hash_res = hash_func(items[height - 1]);
        offset   = order * offset + hash_res;
        leaves[offset].iset_list.insert(iset);
    }

    /** \brief Hash input array of items using tree and return corresponding
        itemset, if found; NULL otherwise */
    assocrules_itemset<cpu> * hash_subset(size_t iset_size, const size_t * subset, int * levelMiss)
    {
        *levelMiss = 0;
        size_t hash_res;
        size_t base = 0, offset = 0;
        hash_res = hash_func(subset[0]);
        if (!is_child(base + offset, hash_res))
        {
            *levelMiss = 1;
            return NULL;
        }
        offset = hash_res;
        base   = 1;
        for (int i = 1; i < height - 1; i++)
        {
            hash_res = hash_func(subset[i]);
            if (!is_child(base + offset, hash_res))
            {
                *levelMiss = i + 1;
                return NULL;
            }
            base += n_nodes_i[i];
            offset = order * offset + hash_res;
        }
        hash_res = hash_func(subset[height - 1]);
        offset   = order * offset + hash_res;

        if (leaves[offset].iset_list.size == 0)
        {
            *levelMiss = height;
            return NULL;
        }

        /* Here if a leaf node was reached */
        for (auto * current = leaves[offset].iset_list.start; current; current = current->next())
        {
            const size_t * curItems = current->itemSet()->items;
            if (!assocrules_memcmp<cpu>(subset, curItems, iset_size))
            {
                return current->itemSet();
            }
        }
        return NULL;
    }

    hash_tree_node<cpu> * root;   /*<! Array of internal nodes */
    hash_tree_leaf<cpu> * leaves; /*<! Array of leaf nodes */
    int * n_nodes_i;              /*<! Number of nodes of depth i */
    int height;                   /*<! Height of the tree */
    int n_nodes;                  /*<! Total number of nodes */
    int n_leaves;                 /*<! Number of leaf nodes */

    int order;    /*<! Tree order. Maximum number of children nodes.
                                  Should be power of 2! */
    int order_m1; /*<! order - 1 */
    int logOrder; /*<! Logarithm of the tree order */
    static const int maxLogOrder = 12;
    static const int minLogOrder = 3;
    static const int maxHeight   = 10;

    bool ok() const { return _status.ok(); }
    services::Status getLastStatus() const { return _status; }

protected:
    services::Status _status;

private:
    hash_tree(const hash_tree &) {};
};

} // namespace internal

} // namespace association_rules

} // namespace algorithms

} // namespace daal

#endif
