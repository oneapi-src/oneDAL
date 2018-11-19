/* file: linear_model_training_batch.h */
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

#ifndef __LINEAR_MODEL_TRAINING_BATCH_H__
#define __LINEAR_MODEL_TRAINING_BATCH_H__

#include "algorithms/linear_model/linear_model_training_types.h"
#include "algorithms/regression/regression_training_batch.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace training
{
namespace interface1
{
/**
 * @defgroup linear_model_training_batch Batch
 * @ingroup linear_model_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__TRAINING__BATCH"></a>
 * \brief Provides methods for linear model model-based training in the batch processing mode
 *
 * \par References
 *      - \ref linear_model::interface1::Model "linear_model::Model" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Batch : public regression::training::Batch
{
public:
    typedef algorithms::linear_model::training::Input  InputType;
    typedef algorithms::linear_model::Parameter        ParameterType;
    typedef algorithms::linear_model::training::Result ResultType;

    /**
     * Returns the structure that contains the result of linear model model-based training
     * \return Structure that contains the result of linear model model-based training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }
};
/** @} */
}
using interface1::Batch;
}
}
}
}
#endif
