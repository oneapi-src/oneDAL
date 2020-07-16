/* file: reducer.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifndef __REDUCER_H__
#define __REDUCER_H__

#include "src/sycl/math_service_types.h"
#include "services/buffer.h"
#include "src/sycl/cl_kernels/sum_reducer.cl"
#include "sycl/internal/types_utils.h"
#include "sycl/internal/execution_context.h"

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace math
{

enum class binary_operation
{
    min,
    max,
    sum,
    sum_of_squares
};

enum class data_layout
{
    row_major,
    col_major
};