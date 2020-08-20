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

#include <vector>
#include <string>
#include <fstream>

inline bool check_file(const std::string& name) {
    return std::ifstream{name}.good();
}

inline std::string get_data_path(const std::string& name) {
    const std::vector<std::string> paths = {
        "../data",
        "examples/oneapi/data"
    };

    for (const auto& path : paths) {
        const std::string try_path = path + "/" + name;
        if (check_file(try_path)) {
            return try_path;
        }
    }

    return name;
}
