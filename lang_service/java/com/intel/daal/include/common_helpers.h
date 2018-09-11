/* file: common_helpers.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include "common_helpers_batch.h"
#include "common_helpers_distributed.h"
#include "common_helpers_online.h"
#include "common_helpers_argument.h"
#include "common_helpers_functions.h"

#define USING_COMMON_NAMESPACES()\
using namespace daal;\
using namespace daal::data_management;\
using namespace daal::algorithms;\
using namespace daal::services;
