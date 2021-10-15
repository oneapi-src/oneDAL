/*******************************************************************************
 * Copyright 2021 Intel Corporation
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

#include <fstream>
#include <string>
#include <vector>

#include "oneapi/dal/detail/spmd_policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

inline bool check_file(const std::string &name) {
  return std::ifstream{name}.good();
}

inline std::string get_data_path(const std::string &name) {
  const std::vector<std::string> paths = {"./data",
                                          "samples/oneapi/dpc/mpi/data"};

  for (const auto &path : paths) {
    const std::string try_path = path + "/" + name;
    if (check_file(try_path)) {
      return try_path;
    }
  }

  return name;
}

template <typename Float>
std::vector<oneapi::dal::table>
split_table_by_rows(const oneapi::dal::detail::data_parallel_policy &p,
                    const oneapi::dal::table &t, std::int64_t split_count) {
  ONEDAL_ASSERT(split_count > 0);
  ONEDAL_ASSERT(split_count <= t.get_row_count());

  const std::int64_t row_count = t.get_row_count();
  const std::int64_t column_count = t.get_column_count();
  const std::int64_t block_size_regular =
      row_count / split_count + bool(row_count % split_count);

  std::vector<oneapi::dal::table> result(split_count);

  for (std::int64_t i = 0; i < split_count; i++) {
    const std::int64_t block_start = i * block_size_regular;
    std::int64_t block_end =
        std::min(block_start + block_size_regular, row_count);
    const std::int64_t block_size = block_end - block_start;

    const auto row_range = oneapi::dal::range{block_start, block_end};
    const auto block = oneapi::dal::row_accessor<const Float>{t}.pull(
        p.get_queue(), row_range, sycl::usm::alloc::device);
    result[i] =
        oneapi::dal::homogen_table::wrap(block, block_size, column_count);
  }

  return result;
}
