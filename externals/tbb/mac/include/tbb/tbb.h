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

#ifndef __TBB_tbb_H
#define __TBB_tbb_H

/**
    This header bulk-includes declarations or definitions of all the functionality
    provided by TBB (save for malloc dependent headers).

    If you use only a few TBB constructs, consider including specific headers only.
    Any header listed below can be included independently of others.
**/

#if TBB_PREVIEW_AGGREGATOR
#include "aggregator.h"
#endif
#include "aligned_space.h"
#include "atomic.h"
#include "blocked_range.h"
#include "blocked_range2d.h"
#include "blocked_range3d.h"
#include "cache_aligned_allocator.h"
#include "combinable.h"
#include "concurrent_hash_map.h"
#if TBB_PREVIEW_CONCURRENT_LRU_CACHE
#include "concurrent_lru_cache.h"
#endif
#include "concurrent_priority_queue.h"
#include "concurrent_queue.h"
#include "concurrent_unordered_map.h"
#include "concurrent_unordered_set.h"
#include "concurrent_vector.h"
#include "critical_section.h"
#include "enumerable_thread_specific.h"
#include "flow_graph.h"
#if TBB_PREVIEW_GLOBAL_CONTROL
#include "global_control.h"
#endif
#include "mutex.h"
#include "null_mutex.h"
#include "null_rw_mutex.h"
#include "parallel_do.h"
#include "parallel_for.h"
#include "parallel_for_each.h"
#include "parallel_invoke.h"
#include "parallel_reduce.h"
#include "parallel_scan.h"
#include "parallel_sort.h"
#include "partitioner.h"
#include "pipeline.h"
#include "queuing_mutex.h"
#include "queuing_rw_mutex.h"
#include "reader_writer_lock.h"
#include "recursive_mutex.h"
#include "spin_mutex.h"
#include "spin_rw_mutex.h"
#include "task.h"
#include "task_arena.h"
#include "task_group.h"
#include "task_scheduler_init.h"
#include "task_scheduler_observer.h"
#include "tbb_allocator.h"
#include "tbb_exception.h"
#include "tbb_thread.h"
#include "tick_count.h"

#endif /* __TBB_tbb_H */
