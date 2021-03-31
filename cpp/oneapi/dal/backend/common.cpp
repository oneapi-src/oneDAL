/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/backend/common.hpp"

namespace oneapi::dal::backend {

INSTANTIATE_TYPE_MAP(float)
INSTANTIATE_TYPE_MAP(double)
INSTANTIATE_TYPE_MAP(std::uint8_t)
INSTANTIATE_TYPE_MAP(std::uint16_t)
INSTANTIATE_TYPE_MAP(std::uint32_t)
INSTANTIATE_TYPE_MAP(std::uint64_t)
INSTANTIATE_TYPE_MAP(std::int8_t)
INSTANTIATE_TYPE_MAP(std::int16_t)
INSTANTIATE_TYPE_MAP(std::int32_t)
INSTANTIATE_TYPE_MAP(std::int64_t)

} // namespace oneapi::dal::backend
