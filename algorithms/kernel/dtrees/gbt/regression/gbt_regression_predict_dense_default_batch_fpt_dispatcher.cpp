/* file: gbt_regression_predict_dense_default_batch_fpt_dispatcher.cpp */
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

/*
//++
//  Implementation of gradient boosted trees algorithm container -- a class
//  that contains fast gradient boosted trees prediction kernels
//  for supported architectures.
//--
*/

#include "gbt_regression_predict_container.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(gbt::regression::prediction::BatchContainer, batch,\
    DAAL_FPTYPE, gbt::regression::prediction::defaultDense)
}
namespace gbt
{
namespace regression
{
namespace prediction
{
namespace interface1
{
template <>
Batch<DAAL_FPTYPE, gbt::regression::prediction::defaultDense>::Batch()
{
    _par = new ParameterType();
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, gbt::regression::prediction::defaultDense>;
template <>
Batch<DAAL_FPTYPE, gbt::regression::prediction::defaultDense>::Batch(const BatchType &other) : input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}
}
}
}
}
}
} // namespace daal
