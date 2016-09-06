/* file: common_helpers.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "common_helpers_batch.h"
#include "common_helpers_distributed.h"
#include "common_helpers_online.h"
#include "common_helpers_argument.h"

#define USING_COMMON_NAMESPACES()\
using namespace daal;\
using namespace daal::data_management;\
using namespace daal::algorithms;\
using namespace daal::services;
