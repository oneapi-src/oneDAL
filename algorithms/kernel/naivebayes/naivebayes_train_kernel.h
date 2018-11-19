/* file: naivebayes_train_kernel.h */
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
//  Declaration of template function that computes K-means.
//--
*/

#ifndef __NAIVEBAYES_TRAIN_FPK_H__
#define __NAIVEBAYES_TRAIN_FPK_H__

#include "multinomial_naive_bayes_model.h"
#include "multinomial_naive_bayes_training_types.h"
#include "kernel.h"
#include "numeric_table.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
namespace training
{
namespace internal
{

template <typename intFPtype, Method method, CpuType cpu>
class NaiveBayesBatchTrainKernel : public Kernel
{
public:
    services::Status compute(const NumericTable *data, const NumericTable *labels, Model *r, const Parameter *par);
};

template <typename intFPtype, Method method, CpuType cpu>
class NaiveBayesOnlineTrainKernel : public Kernel
{
public:
    services::Status compute(const NumericTable *data, const NumericTable *labels, PartialModel *r, const Parameter *par);
    services::Status finalizeCompute(PartialModel *p, Model *r, const Parameter *par);
};

template <typename intFPtype, Method method, CpuType cpu>
class NaiveBayesDistributedTrainKernel : public NaiveBayesOnlineTrainKernel<intFPtype, method, cpu>
{
public:
    services::Status merge(size_t na, PartialModel *const *a, PartialModel *r, const Parameter *par);
};

} // namespace internal
} // namespace training
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal

#endif
