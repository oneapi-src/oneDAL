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

#ifndef __TBB_range_iterator_H
#define __TBB_range_iterator_H

#include "../tbb_stddef.h"

#if __TBB_CPP11_STD_BEGIN_END_PRESENT && __TBB_CPP11_AUTO_PRESENT && __TBB_CPP11_DECLTYPE_PRESENT
    #include <iterator>
#endif

namespace tbb {
    // iterators to first and last elements of container
    namespace internal {

#if __TBB_CPP11_STD_BEGIN_END_PRESENT && __TBB_CPP11_AUTO_PRESENT && __TBB_CPP11_DECLTYPE_PRESENT
        using std::begin;
        using std::end;
        template<typename Container>
        auto first(Container& c)-> decltype(begin(c))  {return begin(c);}

        template<typename Container>
        auto first(const Container& c)-> decltype(begin(c))  {return begin(c);}

        template<typename Container>
        auto last(Container& c)-> decltype(begin(c))  {return end(c);}

        template<typename Container>
        auto last(const Container& c)-> decltype(begin(c)) {return end(c);}
#else
        template<typename Container>
        typename Container::iterator first(Container& c) {return c.begin();}

        template<typename Container>
        typename Container::const_iterator first(const Container& c) {return c.begin();}

        template<typename Container>
        typename Container::iterator last(Container& c) {return c.end();}

        template<typename Container>
        typename Container::const_iterator last(const Container& c) {return c.end();}
#endif

        template<typename T, size_t size>
        T* first(T (&arr) [size]) {return arr;}

        template<typename T, size_t size>
        T* last(T (&arr) [size]) {return arr + size;}
    } //namespace internal
}  //namespace tbb

#endif // __TBB_range_iterator_H
