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

#ifndef __TBB_parallel_for_each_H
#define __TBB_parallel_for_each_H

#include "parallel_do.h"
#include "parallel_for.h"

namespace tbb {

//! @cond INTERNAL
namespace internal {
    // The class calls user function in operator()
    template <typename Function, typename Iterator>
    class parallel_for_each_body_do : internal::no_assign {
        const Function &my_func;
    public:
        parallel_for_each_body_do(const Function &_func) : my_func(_func) {}

        void operator()(typename std::iterator_traits<Iterator>::reference value) const {
            my_func(value);
        }
    };

    // The class calls user function in operator()
    template <typename Function, typename Iterator>
    class parallel_for_each_body_for : internal::no_assign {
        const Function &my_func;
    public:
        parallel_for_each_body_for(const Function &_func) : my_func(_func) {}

        void operator()(tbb::blocked_range<Iterator> range) const {
#if __INTEL_COMPILER
#pragma ivdep
#endif
            for(Iterator it = range.begin(), end = range.end(); it != end; ++it) {
                my_func(*it);
            }
        }
    };

    template<typename Iterator, typename Function, typename Generic>
    struct parallel_for_each_impl {
#if __TBB_TASK_GROUP_CONTEXT
        static void doit(Iterator first, Iterator last, const Function& f, task_group_context &context) {
            internal::parallel_for_each_body_do<Function, Iterator> body(f);
            tbb::parallel_do(first, last, body, context);
        }
#endif
        static void doit(Iterator first, Iterator last, const Function& f) {
            internal::parallel_for_each_body_do<Function, Iterator> body(f);
            tbb::parallel_do(first, last, body);
        }
    };
    template<typename Iterator, typename Function>
    struct parallel_for_each_impl<Iterator, Function, std::random_access_iterator_tag> {
#if __TBB_TASK_GROUP_CONTEXT
        static void doit(Iterator first, Iterator last, const Function& f, task_group_context &context) {
            internal::parallel_for_each_body_for<Function, Iterator> body(f);
            tbb::parallel_for(tbb::blocked_range<Iterator>(first, last), body, context);
        }
#endif
        static void doit(Iterator first, Iterator last, const Function& f) {
            internal::parallel_for_each_body_for<Function, Iterator> body(f);
            tbb::parallel_for(tbb::blocked_range<Iterator>(first, last), body);
        }
    };
} // namespace internal
//! @endcond

/** \name parallel_for_each
    **/
//@{
//! Calls function f for all items from [first, last) interval using user-supplied context
/** @ingroup algorithms */
#if __TBB_TASK_GROUP_CONTEXT
template<typename Iterator, typename Function>
void parallel_for_each(Iterator first, Iterator last, const Function& f, task_group_context &context) {
    internal::parallel_for_each_impl<Iterator, Function, typename std::iterator_traits<Iterator>::iterator_category>::doit(first, last, f, context);
}

//! Calls function f for all items from rng using user-supplied context
/** @ingroup algorithms */
template<typename Range, typename Function>
void parallel_for_each(Range& rng, const Function& f, task_group_context& context) {
    parallel_for_each(tbb::internal::first(rng), tbb::internal::last(rng), f, context);
}

//! Calls function f for all items from const rng user-supplied context
/** @ingroup algorithms */
template<typename Range, typename Function>
void parallel_for_each(const Range& rng, const Function& f, task_group_context& context) {
    parallel_for_each(tbb::internal::first(rng), tbb::internal::last(rng), f, context);
}
#endif /* __TBB_TASK_GROUP_CONTEXT */

//! Uses default context
template<typename Iterator, typename Function>
void parallel_for_each(Iterator first, Iterator last, const Function& f) {
    internal::parallel_for_each_impl<Iterator, Function, typename std::iterator_traits<Iterator>::iterator_category>::doit(first, last, f);
}

//! Uses default context
template<typename Range, typename Function>
void parallel_for_each(Range& rng, const Function& f) {
    parallel_for_each(tbb::internal::first(rng), tbb::internal::last(rng), f);
}

//! Uses default context
template<typename Range, typename Function>
void parallel_for_each(const Range& rng, const Function& f) {
    parallel_for_each(tbb::internal::first(rng), tbb::internal::last(rng), f);
}

//@}

} // namespace

#endif /* __TBB_parallel_for_each_H */
