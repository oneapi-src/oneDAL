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

#ifndef _FGT_TBB_TRACE_IMPL_H
#define _FGT_TBB_TRACE_IMPL_H

#include "../tbb_profiling.h"

namespace tbb {
    namespace internal {

#if TBB_PREVIEW_ALGORITHM_TRACE

        static inline void fgt_algorithm( string_index t, void *algorithm, void *parent ) {
            itt_make_task_group( ITT_DOMAIN_FLOW, algorithm, FGT_ALGORITHM, parent, FGT_ALGORITHM, t );
        }
        static inline void fgt_begin_algorithm( string_index t, void *algorithm ) {
            itt_task_begin( ITT_DOMAIN_FLOW, algorithm, FGT_ALGORITHM, NULL, FLOW_NULL, t );
        }
        static inline void fgt_end_algorithm( void * ) {
            itt_task_end( ITT_DOMAIN_FLOW );
        }
        static inline void fgt_alg_begin_body( string_index t, void *body, void *algorithm ) {
            itt_task_begin( ITT_DOMAIN_FLOW, body, FLOW_BODY, algorithm, FGT_ALGORITHM, t );
        }
        static inline void fgt_alg_end_body( void * ) {
            itt_task_end( ITT_DOMAIN_FLOW );
        }

#else // TBB_PREVIEW_ALGORITHM_TRACE

        static inline void fgt_algorithm( string_index /*t*/, void * /*algorithm*/, void * /*parent*/ ) { }
        static inline void fgt_begin_algorithm( string_index /*t*/, void * /*algorithm*/ ) { }
        static inline void fgt_end_algorithm( void * ) { }
        static inline void fgt_alg_begin_body( string_index /*t*/, void * /*body*/, void * /*algorithm*/ ) { }
        static inline void fgt_alg_end_body( void * ) { }

#endif // TBB_PREVIEW_ALGORITHM_TRACEE

    } // namespace internal
} // namespace tbb

#endif
