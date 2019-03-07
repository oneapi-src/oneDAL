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

#ifndef __TBB_global_control_H
#define __TBB_global_control_H

#if !TBB_PREVIEW_GLOBAL_CONTROL && !__TBB_BUILD
#error Set TBB_PREVIEW_GLOBAL_CONTROL before including global_control.h
#endif

#include "tbb_stddef.h"

namespace tbb {
namespace interface9 {

class global_control {
public:
    enum parameter {
        max_allowed_parallelism,
        thread_stack_size,
        parameter_max // insert new parameters above this point
    };

    global_control(parameter p, size_t value) :
        my_value(value), my_next(NULL), my_param(p) {
        __TBB_ASSERT(my_param < parameter_max, "Invalid parameter");
#if __TBB_WIN8UI_SUPPORT
        // For Windows Store* apps it's impossible to set stack size
        if (p==thread_stack_size)
            return;
#elif __TBB_x86_64 && (_WIN32 || _WIN64)
        if (p==thread_stack_size)
            __TBB_ASSERT_RELEASE((unsigned)value == value, "Stack size is limited to unsigned int range");
#endif
        if (my_param==max_allowed_parallelism)
            __TBB_ASSERT_RELEASE(my_value>0, "max_allowed_parallelism cannot be 0.");
        internal_create();
    }

    ~global_control() {
        __TBB_ASSERT(my_param < parameter_max, "Invalid parameter. Probably the object was corrupted.");
#if __TBB_WIN8UI_SUPPORT
        // For Windows Store* apps it's impossible to set stack size
        if (my_param==thread_stack_size)
            return;
#endif
        internal_destroy();
    }

    static size_t active_value(parameter p) {
        __TBB_ASSERT(p < parameter_max, "Invalid parameter");
        return active_value((int)p);
    }
private:
    size_t    my_value;
    global_control *my_next;
    parameter my_param;

    void __TBB_EXPORTED_METHOD internal_create();
    void __TBB_EXPORTED_METHOD internal_destroy();
    static size_t __TBB_EXPORTED_FUNC active_value(int param);
};
} // namespace interface9

using interface9::global_control;

} // tbb

#endif // __TBB_global_control_H
