/*
    Copyright 2005-2017 Intel Corporation.

    The source code, information and material ("Material") contained herein is owned by
    Intel Corporation or its suppliers or licensors, and title to such Material remains
    with Intel Corporation or its suppliers or licensors. The Material contains
    proprietary information of Intel or its suppliers and licensors. The Material is
    protected by worldwide copyright laws and treaty provisions. No part of the Material
    may be used, copied, reproduced, modified, published, uploaded, posted, transmitted,
    distributed or disclosed in any way without Intel's prior express written permission.
    No license under any patent, copyright or other intellectual property rights in the
    Material is granted to or conferred upon you, either expressly, by implication,
    inducement, estoppel or otherwise. Any license under such intellectual property
    rights must be express and approved by Intel in writing.

    Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
    or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
    in any way.
*/

/* Container implementations in this header are based on PPL implementations
   provided by Microsoft. */

#ifndef __TBB_concurrent_unordered_set_H
#define __TBB_concurrent_unordered_set_H

#include "internal/_concurrent_unordered_impl.h"

namespace tbb
{

namespace interface5 {

// Template class for hash set traits
template<typename Key, typename Hash_compare, typename Allocator, bool Allow_multimapping>
class concurrent_unordered_set_traits
{
protected:
    typedef Key value_type;
    typedef Key key_type;
    typedef Hash_compare hash_compare;
    typedef typename Allocator::template rebind<value_type>::other allocator_type;
    enum { allow_multimapping = Allow_multimapping };

    concurrent_unordered_set_traits() : my_hash_compare() {}
    concurrent_unordered_set_traits(const hash_compare& hc) : my_hash_compare(hc) {}

    static const Key& get_key(const value_type& value) {
        return value;
    }

    hash_compare my_hash_compare; // the comparator predicate for keys
};

template <typename Key, typename Hasher = tbb::tbb_hash<Key>, typename Key_equality = std::equal_to<Key>, typename Allocator = tbb::tbb_allocator<Key> >
class concurrent_unordered_set : public internal::concurrent_unordered_base< concurrent_unordered_set_traits<Key, internal::hash_compare<Key, Hasher, Key_equality>, Allocator, false> >
{
    // Base type definitions
    typedef internal::hash_compare<Key, Hasher, Key_equality> hash_compare;
    typedef concurrent_unordered_set_traits<Key, hash_compare, Allocator, false> traits_type;
    typedef internal::concurrent_unordered_base< traits_type > base_type;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using traits_type::allow_multimapping;
public:
    using base_type::insert;

    // Type definitions
    typedef Key key_type;
    typedef typename base_type::value_type value_type;
    typedef Key mapped_type;
    typedef Hasher hasher;
    typedef Key_equality key_equal;
    typedef hash_compare key_compare;

    typedef typename base_type::allocator_type allocator_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;

    typedef typename base_type::size_type size_type;
    typedef typename base_type::difference_type difference_type;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::iterator local_iterator;
    typedef typename base_type::const_iterator const_local_iterator;

    // Construction/destruction/copying
    explicit concurrent_unordered_set(size_type n_of_buckets = base_type::initial_bucket_number, const hasher& a_hasher = hasher(),
        const key_equal& a_keyeq = key_equal(), const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {}

    explicit concurrent_unordered_set(const Allocator& a) : base_type(base_type::initial_bucket_number, key_compare(), a)
    {}

    template <typename Iterator>
    concurrent_unordered_set(Iterator first, Iterator last, size_type n_of_buckets = base_type::initial_bucket_number, const hasher& a_hasher = hasher(),
        const key_equal& a_keyeq = key_equal(), const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        insert(first, last);
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_unordered_set(std::initializer_list<value_type> il, size_type n_of_buckets = base_type::initial_bucket_number, const hasher& a_hasher = hasher(),
        const key_equal& a_keyeq = key_equal(), const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        this->insert(il.begin(),il.end());
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT
#if !__TBB_IMPLICIT_MOVE_PRESENT
    concurrent_unordered_set(const concurrent_unordered_set& table)
        : base_type(table)
    {}

    concurrent_unordered_set& operator=(const concurrent_unordered_set& table)
    {
        return static_cast<concurrent_unordered_set&>(base_type::operator=(table));
    }

    concurrent_unordered_set(concurrent_unordered_set&& table)
        : base_type(std::move(table))
    {}

    concurrent_unordered_set& operator=(concurrent_unordered_set&& table)
    {
        return static_cast<concurrent_unordered_set&>(base_type::operator=(std::move(table)));
    }
#endif //!__TBB_IMPLICIT_MOVE_PRESENT

    concurrent_unordered_set(concurrent_unordered_set&& table, const Allocator& a)
        : base_type(std::move(table), a)
    {}
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

    concurrent_unordered_set(const concurrent_unordered_set& table, const Allocator& a)
        : base_type(table, a)
    {}

};

template <typename Key, typename Hasher = tbb::tbb_hash<Key>, typename Key_equality = std::equal_to<Key>,
         typename Allocator = tbb::tbb_allocator<Key> >
class concurrent_unordered_multiset :
    public internal::concurrent_unordered_base< concurrent_unordered_set_traits<Key,
    internal::hash_compare<Key, Hasher, Key_equality>, Allocator, true> >
{
    // Base type definitions
    typedef internal::hash_compare<Key, Hasher, Key_equality> hash_compare;
    typedef concurrent_unordered_set_traits<Key, hash_compare, Allocator, true> traits_type;
    typedef internal::concurrent_unordered_base< traits_type > base_type;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using traits_type::allow_multimapping;
public:
    using base_type::insert;

    // Type definitions
    typedef Key key_type;
    typedef typename base_type::value_type value_type;
    typedef Key mapped_type;
    typedef Hasher hasher;
    typedef Key_equality key_equal;
    typedef hash_compare key_compare;

    typedef typename base_type::allocator_type allocator_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;

    typedef typename base_type::size_type size_type;
    typedef typename base_type::difference_type difference_type;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::iterator local_iterator;
    typedef typename base_type::const_iterator const_local_iterator;

    // Construction/destruction/copying
    explicit concurrent_unordered_multiset(size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {}

    explicit concurrent_unordered_multiset(const Allocator& a) : base_type(base_type::initial_bucket_number, key_compare(), a)
    {}

    template <typename Iterator>
    concurrent_unordered_multiset(Iterator first, Iterator last, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {
        insert(first, last);
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_unordered_multiset(std::initializer_list<value_type> il, size_type n_of_buckets = base_type::initial_bucket_number, const hasher& a_hasher = hasher(),
        const key_equal& a_keyeq = key_equal(), const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        this->insert(il.begin(),il.end());
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT
#if !__TBB_IMPLICIT_MOVE_PRESENT
    concurrent_unordered_multiset(const concurrent_unordered_multiset& table)
        : base_type(table)
    {}

    concurrent_unordered_multiset& operator=(const concurrent_unordered_multiset& table)
    {
        return static_cast<concurrent_unordered_multiset&>(base_type::operator=(table));
    }

    concurrent_unordered_multiset(concurrent_unordered_multiset&& table)
        : base_type(std::move(table))
    {}

    concurrent_unordered_multiset& operator=(concurrent_unordered_multiset&& table)
    {
        return static_cast<concurrent_unordered_multiset&>(base_type::operator=(std::move(table)));
    }
#endif //!__TBB_IMPLICIT_MOVE_PRESENT

    concurrent_unordered_multiset(concurrent_unordered_multiset&& table, const Allocator& a)
        : base_type(std::move(table), a)
    {
    }
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

    concurrent_unordered_multiset(const concurrent_unordered_multiset& table, const Allocator& a)
        : base_type(table, a)
    {}
};
} // namespace interface5

using interface5::concurrent_unordered_set;
using interface5::concurrent_unordered_multiset;

} // namespace tbb

#endif// __TBB_concurrent_unordered_set_H
