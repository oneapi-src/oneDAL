/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>

namespace oneapi::dal::network {

namespace detail {

class communicator_base {
public:
    virtual void allreduce(float* ptr, size_t n) = 0;    
    virtual void allreduce(double* ptr, size_t n) = 0;    
    virtual void allreduce(int* ptr, size_t n) = 0;    
};

class empty_communicator: public communicator_base {
public:
    void allreduce(float* ptr, size_t n) override { }
    void allreduce(double* ptr, size_t n) override { }
    void allreduce(int* ptr, size_t n) override { }
};

}

class network {
public:
    // TODO: make it private for users and public for DAAL: friend fucntion/private in public class and public in base class
    virtual std::shared_ptr<detail::communicator_base> get_communicator() const = 0;
};


class empty_network: public network {
public:
    virtual std::shared_ptr<detail::communicator_base> get_communicator() const
    {
        return std::shared_ptr<detail::communicator_base>(new detail::empty_communicator() );
    }
};

}

