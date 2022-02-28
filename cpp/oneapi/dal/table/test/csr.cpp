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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/test/engine/common.hpp"


namespace oneapi::dal::test {

TEST("can construct empty table") {
    csr_table t;

    REQUIRE(t.has_data() == false);
    REQUIRE(t.get_kind() == csr_table::kind());
    REQUIRE(t.get_row_count() == 0);
    REQUIRE(t.get_column_count() == 0);
}

} // namespace oneapi::dal::test
