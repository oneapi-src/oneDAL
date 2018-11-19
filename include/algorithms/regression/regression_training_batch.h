/* file: regression_training_batch.h */
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
//  Implementation of the interface for the regression model-based training
//  in the batch processing mode
//--
*/

#ifndef __REGRESSION_TRAINING_BATCH_H__
#define __REGRESSION_TRAINING_BATCH_H__

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
 * @defgroup base_regression_training_batch Batch
 * @ingroup base_regression_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__REGRESSION__TRAINING__BATCH"></a>
 * \brief Provides methods for the regression model-based training in the batch processing mode
 *
 * \par References
 *      - \ref regression::interface1::Model "regression::Model" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Batch : public Training<batch>
{
public:
    typedef algorithms::regression::training::Input  InputType;
    typedef algorithms::regression::training::Result ResultType;

    virtual ~Batch() {}
    /**
     * Get input objects for the regression model-based training algorithm
     * \return Input objects for the regression model-based training algorithm
     */
    virtual InputType* getInput() = 0;

    /**
     * Registers user-allocated memory to store the result of the regression model-based training
     * \param[in] res    Structure to store the result of the regression model-based training
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
     * Returns the structure that contains the result of the regression model-based training
     * \return Structure that contains the result of the regression model-based training
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
