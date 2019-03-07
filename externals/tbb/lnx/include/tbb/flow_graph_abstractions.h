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

#ifndef __TBB_flow_graph_abstractions_H
#define __TBB_flow_graph_abstractions_H

namespace tbb {
namespace flow {
namespace interface10 {

//! Pure virtual template classes that define interfaces for async communication
class graph_proxy {
public:
    //! Inform a graph that messages may come from outside, to prevent premature graph completion
    virtual void reserve_wait() = 0;

    //! Inform a graph that a previous call to reserve_wait is no longer in effect
    virtual void release_wait() = 0;

    virtual ~graph_proxy() {}
};

template <typename Input>
class receiver_gateway : public graph_proxy {
public:
    //! Type of inputing data into FG.
    typedef Input input_type;

    //! Submit signal from an asynchronous activity to FG.
    virtual bool try_put(const input_type&) = 0;
};

} //interfaceX

using interface10::graph_proxy;
using interface10::receiver_gateway;

} //flow
} //tbb
#endif
