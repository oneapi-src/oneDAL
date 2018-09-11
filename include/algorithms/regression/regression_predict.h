/* file: regression_predict.h */
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
//  Implementation of the interface for the regression model-based prediction
//--
*/

#ifndef __REGRESSION_PREDICT_H__
#define __REGRESSION_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/regression/regression_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace regression
{
namespace prediction
{
namespace interface1
{
/**
 * @defgroup base_regression_prediction_batch Batch
 * @ingroup base_regression_prediction
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__PREDICTION__BATCH"></a>
 * \brief Provides methods to run implementations of the regression model-based prediction
 *
 * \par References
 *      - \ref regression::interface1::Model "regression::Model" class
 *      - \ref training::interface1::Batch "training::Batch" class
 */
class Batch : public daal::algorithms::Prediction
{
public:
    typedef algorithms::regression::prediction::Input  InputType;
    typedef algorithms::Parameter                      ParameterType;
    typedef algorithms::regression::prediction::Result ResultType;

    virtual ~Batch() {}
    virtual InputType* getInput() = 0;

    /**
     * Registers user-allocated memory to store the result of the regression model-based prediction
     * \param[in] res    Structure to store the result of the regression model-based prediction
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns the structure that contains the result of the regression model-based prediction
     * \return Structure that contains the result of the the regression model-based prediction
     */
    ResultPtr getResult() { return _result; }

protected:
    ResultPtr _result;
};
/** @} */
}
using interface1::Batch;
}
}
}
}
#endif
