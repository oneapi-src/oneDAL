/* file: linear_regression_model_builder_fpt.cpp */
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

#include "algorithms/linear_regression/linear_regression_model_builder.h"
#include "linear_regression_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace interface1
{
using namespace daal::data_management;

template<typename modelFPType>
ModelBuilder<modelFPType>::ModelBuilder(size_t nFeatures, size_t nResponses): _nFeatures(nFeatures), _nResponses(nResponses)
{
    Parameter* p = new Parameter;
    p->interceptFlag = true; /* default intercept flag is true but can be changed via setInterceptFlag */
    modelFPType dummy = 1.0;
    _modelPtr = linear_regression::ModelPtr(new linear_regression::internal::ModelImpl(_nFeatures, _nResponses, *p, dummy));
}

template<typename modelFPType>
void ModelBuilder<modelFPType>::setInterceptFlag(bool interceptFlag)
{
    linear_regression::internal::ModelImpl* m = static_cast<linear_regression::internal::ModelImpl*>(_modelPtr.get());
    m->setInterceptFlag(interceptFlag);
}

template class ModelBuilder<DAAL_FPTYPE>;
}
}
}
}
