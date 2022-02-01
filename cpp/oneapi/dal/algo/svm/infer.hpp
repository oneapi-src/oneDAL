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

#include "oneapi/dal/algo/svm/detail/infer_ops.hpp"
#include "oneapi/dal/algo/svm/infer_types.hpp"
#include "oneapi/dal/infer.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename Descriptor>
struct infer_ops<Descriptor, dal::svm::detail::descriptor_tag>
        : dal::svm::detail::infer_ops<Descriptor> {};

} // namespace v1
} // namespace oneapi::dal::detail
