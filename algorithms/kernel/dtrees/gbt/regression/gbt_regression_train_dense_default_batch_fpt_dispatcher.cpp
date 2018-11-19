/* file: gbt_regression_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of gradient boosted trees container.
//--
*/

#include "gbt_regression_train_container.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(gbt::regression::training::BatchContainer, batch, DAAL_FPTYPE, \
    gbt::regression::training::defaultDense)
}

namespace gbt
{
namespace regression
{
namespace training
{
namespace interface1
{
template <>
Batch<DAAL_FPTYPE, gbt::regression::training::defaultDense>::Batch()
{
    _par = new ParameterType();
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, gbt::regression::training::defaultDense>;
template <>
Batch<DAAL_FPTYPE, gbt::regression::training::defaultDense>::Batch(const BatchType &other) : input(other.input)
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
