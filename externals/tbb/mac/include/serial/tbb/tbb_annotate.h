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

#ifndef __TBB_annotate_H
#define __TBB_annotate_H

// Macros used by the Intel(R) Parallel Advisor.
#ifdef __TBB_NORMAL_EXECUTION
    #define ANNOTATE_SITE_BEGIN( site )
    #define ANNOTATE_SITE_END( site )
    #define ANNOTATE_TASK_BEGIN( task )
    #define ANNOTATE_TASK_END( task )
    #define ANNOTATE_LOCK_ACQUIRE( lock )
    #define ANNOTATE_LOCK_RELEASE( lock )
#else
    #include <advisor-annotate.h>
#endif

#endif /* __TBB_annotate_H */
