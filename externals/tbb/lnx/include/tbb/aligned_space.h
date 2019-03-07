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

#ifndef __TBB_aligned_space_H
#define __TBB_aligned_space_H

#include "tbb_stddef.h"
#include "tbb_machine.h"

namespace tbb {

//! Block of space aligned sufficiently to construct an array T with N elements.
/** The elements are not constructed or destroyed by this class.
    @ingroup memory_allocation */
template<typename T,size_t N=1>
class aligned_space {
private:
    typedef __TBB_TypeWithAlignmentAtLeastAsStrict(T) element_type;
    element_type array[(sizeof(T)*N+sizeof(element_type)-1)/sizeof(element_type)];
public:
    //! Pointer to beginning of array
    T* begin() const {return internal::punned_cast<T*>(this);}

    //! Pointer to one past last element in array.
    T* end() const {return begin()+N;}
};

} // namespace tbb

#endif /* __TBB_aligned_space_H */
