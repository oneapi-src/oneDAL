/* file: naivebayes_predict_csr_fast_fpt_cpu.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of multinomial naive bayes algorithm.
//--
*/

#include "src/algorithms/naivebayes/naivebayes_predict_kernel.h"
#include "src/algorithms/naivebayes/naivebayes_predict_fast_impl.i"
#include "src/algorithms/naivebayes/naivebayes_predict_container.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace prediction
{
template class BatchContainer<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
namespace internal
{
template class NaiveBayesPredictKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
} // namespace internal
} // namespace prediction
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
