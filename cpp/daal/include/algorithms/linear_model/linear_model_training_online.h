/* file: linear_model_training_online.h */
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

#ifndef __LINEAR_MODEL_TRAINING_ONLINE_H__
#define __LINEAR_MODEL_TRAINING_ONLINE_H__

#include "algorithms/linear_model/linear_model_training_types.h"
#include "algorithms/regression/regression_training_online.h"

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
 * @defgroup linear_model_training_online Online
 * @ingroup linear_model_training
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_MODEL__TRAINING__ONLINE"></a>
 * \brief Provides methods for the linear model-based training in the online processing mode
 *
 * \par References
 *      - \ref linear_model::interface1::Model "linear_model::Model" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Online : public regression::training::Online
{
public:
    typedef algorithms::linear_model::training::Input InputType;
    typedef algorithms::linear_model::Parameter ParameterType;
    typedef algorithms::linear_model::training::Result ResultType;
    typedef algorithms::linear_model::training::PartialResult PartialResultType;

    /**
     * Returns the structure that contains a partial result of the linear model-based training
     * \return Structure that contains a partial result of the linear model-based training
     */
    PartialResultPtr getPartialResult() { return PartialResultType::cast(_partialResult); }

    /**
     * Returns the structure that contains the result of the linear model-based training
     * \return Structure that contains the result of the linear model-based training
     */
    ResultPtr getResult() { return ResultType::cast(_result); }
}; // class  : public Online
/** @} */
} // namespace interface1
using interface1::Online;

} // namespace training
} // namespace linear_model
} // namespace algorithms
} // namespace daal
#endif
