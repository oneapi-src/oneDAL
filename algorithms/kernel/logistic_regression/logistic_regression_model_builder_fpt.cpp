/* file: logistic_regression_model_builder_fpt.cpp */
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

#include "algorithms/logistic_regression/logistic_regression_model_builder.h"
#include "logistic_regression_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace interface1
{
using namespace daal::data_management;

template<typename modelFPType>
ModelBuilder<modelFPType>::ModelBuilder(size_t nFeatures, size_t nClasses) : _nFeatures(nFeatures), _nClasses(nClasses)
{
    const bool interceptFlag = true; /* default intercept flag is true but can be changed via setInterceptFlag */
    modelFPType dummy = 1.0;
    _modelPtr = logistic_regression::ModelPtr(new logistic_regression::internal::ModelImpl(_nFeatures, interceptFlag, _nClasses, dummy, &_s));
}

template<typename modelFPType>
void ModelBuilder<modelFPType>::setInterceptFlag(bool interceptFlag)
{
    logistic_regression::internal::ModelImpl* const m = static_cast<logistic_regression::internal::ModelImpl*>(_modelPtr.get());
    _s = m->reset(interceptFlag);
}

template class ModelBuilder<DAAL_FPTYPE>;
}// namespace interface1
}// namespace logistic_regression
}// namespace algorithms
}// namespace daal
