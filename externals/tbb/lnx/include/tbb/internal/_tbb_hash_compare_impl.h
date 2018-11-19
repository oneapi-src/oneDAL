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

// must be included outside namespaces.
#ifndef __TBB_tbb_hash_compare_impl_H
#define __TBB_tbb_hash_compare_impl_H

#include <string>

namespace tbb {
namespace interface5 {
namespace internal {

// Template class for hash compare
template<typename Key, typename Hasher, typename Key_equality>
class hash_compare
{
public:
    typedef Hasher hasher;
    typedef Key_equality key_equal;

    hash_compare() {}

    hash_compare(Hasher a_hasher) : my_hash_object(a_hasher) {}

    hash_compare(Hasher a_hasher, Key_equality a_keyeq) : my_hash_object(a_hasher), my_key_compare_object(a_keyeq) {}

    size_t operator()(const Key& key) const {
        return ((size_t)my_hash_object(key));
    }

    bool operator()(const Key& key1, const Key& key2) const {
        // TODO: get rid of the result invertion
        return (!my_key_compare_object(key1, key2));
    }

    Hasher       my_hash_object;        // The hash object
    Key_equality my_key_compare_object; // The equality comparator object
};

//! Hash multiplier
static const size_t hash_multiplier = tbb::internal::select_size_t_constant<2654435769U, 11400714819323198485ULL>::value;

} // namespace internal

//! Hasher functions
template<typename T>
inline size_t tbb_hasher( const T& t ) {
    return static_cast<size_t>( t ) * internal::hash_multiplier;
}
template<typename P>
inline size_t tbb_hasher( P* ptr ) {
    size_t const h = reinterpret_cast<size_t>( ptr );
    return (h >> 3) ^ h;
}
template<typename E, typename S, typename A>
inline size_t tbb_hasher( const std::basic_string<E,S,A>& s ) {
    size_t h = 0;
    for( const E* c = s.c_str(); *c; ++c )
        h = static_cast<size_t>(*c) ^ (h * internal::hash_multiplier);
    return h;
}
template<typename F, typename S>
inline size_t tbb_hasher( const std::pair<F,S>& p ) {
    return tbb_hasher(p.first) ^ tbb_hasher(p.second);
}

} // namespace interface5
using interface5::tbb_hasher;

// Template class for hash compare
template<typename Key>
class tbb_hash
{
public:
    tbb_hash() {}

    size_t operator()(const Key& key) const
    {
        return tbb_hasher(key);
    }
};

//! hash_compare that is default argument for concurrent_hash_map
template<typename Key>
struct tbb_hash_compare {
    static size_t hash( const Key& a ) { return tbb_hasher(a); }
    static bool equal( const Key& a, const Key& b ) { return a == b; }
};

}  // namespace tbb
#endif  /*  __TBB_tbb_hash_compare_impl_H */
