/* file: regression_training_online.h */
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
//  Implementation of the interface for linear regression model-based training
//  in the online processing mode
//--
*/

#ifndef __REGRESSION_TRAINING_ONLINE_H__
#define __REGRESSION_TRAINING_ONLINE_H__

#include "algorithms/regression/regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace training
{
namespace interface1
{
/**
 * @defgroup base_regression_training_online Online
 * @ingroup base_regression_training
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__TRAINING__ONLINE"></a>
 * \brief Provides methods for the regression model-based training in the online processing mode
 *
 * \par References
 *      - \ref regression::interface1::Model "regression::Model" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Online : public Training<online>
{
public:
    typedef algorithms::regression::training::Input InputType;
    typedef algorithms::regression::training::Result ResultType;
    typedef algorithms::regression::training::PartialResult PartialResultType;

    virtual ~Online() {}
    virtual InputType * getInput() = 0;

    /**
     * Registers user-allocated memory to store a partial result of the regression model-based training
     * \param[in] partialResult    Structure to store a partial result of the regression model-based training
     *
     * \return Status of computations
     */
    services::Status setPartialResult(const PartialResultPtr & partialResult)
    {
        _partialResult = partialResult;
        _pres          = _partialResult.get();
        return services::Status();
    }

    /**
     * Registers user-allocated memory to store the result of the regression model-based training
     * \param[in] res    Structure to store the result of the regression model-based training
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr & res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res    = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains a partial result of the regression model-based training
     * \return Structure that contains a partial result of the regression model-based training
     */
    PartialResultPtr getPartialResult() { return _partialResult; }

    /**
     * Returns the structure that contains the result of the regression model-based training
     * \return Structure that contains the result of the regression model-based training
     */
    ResultPtr getResult() { return _result; }

protected:
    PartialResultPtr _partialResult;
    ResultPtr _result;
}; // class  : public Online
/** @} */
} // namespace interface1
using interface1::Online;

} // namespace training
} // namespace regression
} // namespace algorithms
} // namespace daal
#endif
