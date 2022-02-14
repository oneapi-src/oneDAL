/* file: linear_regression_ne_model.h */
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
//  Declaration of the linear regression model class for the normal equations method
//--
*/

#ifndef __LINREG_NE_MODEL_H__
#define __LINREG_NE_MODEL_H__

#include "algorithms/linear_regression/linear_regression_model.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace interface1
{
/**
 * @ingroup linear_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__MODELNORMEQ"></a>
 * \brief %Model trained with the linear regression algorithm using the normal equations method
 *
 * \par References
 *      - Parameter class
 *      - Prediction class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT ModelNormEq : public Model
{
public:
    DECLARE_MODEL(ModelNormEq, linear_regression::Model);

    virtual ~ModelNormEq() {}

    /**
     * Returns a Numeric table that contains partial sums X'*X
     * \return Numeric table that contains partial sums X'*X
     */
    virtual data_management::NumericTablePtr getXTXTable() = 0;

    /**
     * Returns a Numeric table that contains partial sums X'*Y
     * \return Numeric table that contains partial sums X'*Y
     */
    virtual data_management::NumericTablePtr getXTYTable() = 0;
};
typedef services::SharedPtr<ModelNormEq> ModelNormEqPtr;
typedef services::SharedPtr<const ModelNormEq> ModelNormEqConstPtr;
/** @} */
} // namespace interface1
using interface1::ModelNormEq;
using interface1::ModelNormEqPtr;
using interface1::ModelNormEqConstPtr;

} // namespace linear_regression
} // namespace algorithms
} // namespace daal
#endif
