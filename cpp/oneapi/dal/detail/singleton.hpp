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

namespace oneapi::dal::detail {

namespace v1 {

template <class T>
class singleton {
public:
    static T& get() {
        static std::once_flag flag;
        std::call_once(flag, [] {
            get_instance();
        });
        return get_instance();
    }

private:
    static T& get_instance() {
        static T instance;
        return instance;
    }
};

} // namespace v1

using v1::singleton;

} // namespace oneapi::dal::detail
