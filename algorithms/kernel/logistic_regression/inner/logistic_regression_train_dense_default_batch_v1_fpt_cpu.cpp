/* file: logistic_regression_train_dense_default_batch_v1_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/*
//++
//  Implementation of logistic regression classification training functions for the default method
//--
*/

#include "logistic_regression_train_container_v1.h"
#include "logistic_regression_train_dense_default_v1_impl.i"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace training
{
namespace interface1
{
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}

namespace internal
{
template class I1TrainBatchKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
}

} // namespace training
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
