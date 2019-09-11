/* file: implicit_als_train_utils.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

#ifndef __IMPLICIT_ALS_TRAIN_UTILS_H__
#define __IMPLICIT_ALS_TRAIN_UTILS_H__

#include "services/env_detect.h"
#include "error_handling.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
services::Status csr2csc(size_t nItems, size_t nUsers,
            const algorithmFPType *csrdata, const size_t *colIndices, const size_t *rowOffsets,
            algorithmFPType *cscdata, size_t *rowIndices, size_t *colOffsets);
}
}
}
}
}
#endif
