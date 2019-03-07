/* file: logitboost_train_friedman_fpt_cpu.cpp */
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
//  Implementation of Logit Boost training functions.
//--
*/

#include "logitboost_train_batch_container.h"
#include "logitboost_train_friedman_kernel.h"
#include "logitboost_train_friedman_impl.i"

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, friedman, DAAL_CPU>;
}
namespace internal
{
template struct LogitBoostTrainKernel<friedman, DAAL_FPTYPE, DAAL_CPU>;
}
}
}
}
}
