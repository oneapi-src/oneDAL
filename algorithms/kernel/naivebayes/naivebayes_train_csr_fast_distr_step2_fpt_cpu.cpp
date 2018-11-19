/* file: naivebayes_train_csr_fast_distr_step2_fpt_cpu.cpp */
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
//  Implementation of multinomial naive bayes algorithm.
//--
*/

#include "naivebayes_train_kernel.h"
#include "naivebayes_train_impl.i"
#include "naivebayes_train_container.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
namespace interface1
{
template class DistributedContainer<step2Master, DAAL_FPTYPE, fastCSR, DAAL_CPU>;
}
namespace internal
{
template class NaiveBayesDistributedTrainKernel<DAAL_FPTYPE, fastCSR, DAAL_CPU>;
} // namespace internal
} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
