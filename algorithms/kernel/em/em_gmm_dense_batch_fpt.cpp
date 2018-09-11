/* file: em_gmm_dense_batch_fpt.cpp */
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
//  Implementation of EM Batch constructor
//--
*/

#include "em_gmm.h"

namespace daal
{
namespace algorithms
{
namespace em_gmm
{
namespace interface1
{

template<typename algorithmFPType, Method method>
Batch<algorithmFPType, method>::Batch(const size_t nComponents) :
    parameter(nComponents, services::SharedPtr<covariance::Batch<algorithmFPType, covariance::defaultDense> >
              (new covariance::Batch<algorithmFPType, covariance::defaultDense>()))
{
    initialize();
}

template<typename algorithmFPType, Method method>
void Batch<algorithmFPType, method>::initialize()
{
    Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
    _in = &input;
    _par = &parameter;
    _result = ResultPtr(new Result());
}

template class Batch<DAAL_FPTYPE, defaultDense>;

}
} // namespace em_gmm
} // namespace algorithms
} // namespace daal
