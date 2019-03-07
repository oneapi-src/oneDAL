/* file: implicit_als_predict_ratings_dense_default_kernel.h */
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

/*
//++
//  Declaration of structure containing kernels for implicit ALS
//  prediction.
//--
*/

#ifndef __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_KERNEL_H__
#define __IMPLICIT_ALS_PREDICT_RATINGS_DENSE_DEFAULT_KERNEL_H__

#include "implicit_als_predict_ratings_batch.h"
#include "implicit_als_predict_ratings_distributed.h"
#include "implicit_als_model.h"
#include "kernel.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
class ImplicitALSPredictKernel : public daal::algorithms::Kernel
{
public:
    ImplicitALSPredictKernel() {}
    virtual ~ImplicitALSPredictKernel() {}

    services::Status compute(const NumericTable *usersFactorsTable, const NumericTable *itemsFactorsTable,
                NumericTable *ratingsTable, const Parameter *parameter);
};

}
}
}
}
}
}

#endif
