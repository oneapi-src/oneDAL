/* file: logistic_regression_model.cpp */
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
//  Implementation of the class defining the logistic regression model
//--
*/

#include "logistic_regression_model_impl.h"
#include "serialization_utils.h"
#include "service_numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::logistic_regression::internal;

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{

namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS2(Model, internal::ModelImpl, SERIALIZATION_LOGISTIC_REGRESSION_MODEL_ID);
}

namespace internal
{

ModelImpl::ModelImpl(size_t nFeatures, bool interceptFlag) : ClassificationImplType(nFeatures), _interceptFlag(interceptFlag)
{
}

size_t ModelImpl::getNumberOfBetas() const
{
    return _beta.get() ? _beta->getNumberOfColumns() : 0;
}

bool ModelImpl::getInterceptFlag() const
{
    return _interceptFlag;
}

data_management::NumericTablePtr ModelImpl::getBeta()
{
    return _beta;
}

const data_management::NumericTablePtr ModelImpl::getBeta() const
{
    return _beta;
}

services::Status ModelImpl::reset(bool interceptFlag)
{
    _interceptFlag = interceptFlag;
    const size_t nRows = _beta->getNumberOfRows();
    daal::internal::WriteOnlyRows<float, sse2> rows(*_beta, 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(rows);
    float *ar = rows.get();
    const size_t nCols = _beta->getNumberOfColumns();
    for(size_t i = 0; i < nCols * nRows; i++)
        ar[i] = 0.0f;
    return services::Status();
}

services::Status ModelImpl::serializeImpl(data_management::InputDataArchive  * arch)
{
    auto s = algorithms::classifier::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    return s.add(this->serialImpl<data_management::InputDataArchive, false>(arch));
}

services::Status ModelImpl::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    auto s = algorithms::classifier::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    return s.add(this->serialImpl<const data_management::OutputDataArchive, true>(arch));
}

logistic_regression::ModelPtr ModelImpl::create(size_t nFeatures, bool interceptFlag, services::Status *stat)
{
    logistic_regression::ModelPtr pRes(new logistic_regression::internal::ModelImpl(nFeatures, interceptFlag));
    if((!pRes.get()) && stat)
        stat->add(services::ErrorMemoryAllocationFailed);
    return pRes;
}

} // namespace internal
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal
