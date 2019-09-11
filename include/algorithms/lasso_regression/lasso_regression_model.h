/* file: lasso_regression_model.h */
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
//  Implementation of the class defining the lasso regression model
//--
*/

#ifndef __LASSO_REGRESSION_MODEL_H__
#define __LASSO_REGRESSION_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/linear_model/linear_model_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup lasso_regression LASSO Regression
 * \copydoc daal::algorithms::lasso_regression
 * @ingroup linear_model
 */
namespace lasso_regression
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LASSO_REGRESSION__MODEL"></a>
 * \brief %Base class for models trained with the lasso regression algorithm
 *
 * \tparam modelFPType  Data type to store lasso regression model data, double or float
 *
 * \par References
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public linear_model::Model
{
public:
    DECLARE_MODEL(Model, linear_model::Model);
};
typedef services::SharedPtr<Model> ModelPtr;
typedef services::SharedPtr<const Model> ModelConstPtr;
/** @} */
} // namespace interface1

using interface1::Model;
using interface1::ModelPtr;
using interface1::ModelConstPtr;

/**
 * Checks the correctness of lasso regression model
 * \param[in]  model             The model to check
 * \param[in]  par               The parameter of lasso regression algorithm
 * \param[in]  nBeta             Required number of lasso regression coefficients
 * \param[in]  nResponses        Required number of responses on the training stage
 * \param[in]  method            Computation method
 *
 * \return Status of computations
 */
DAAL_EXPORT services::Status checkModel(
    lasso_regression::Model* model, const daal::algorithms::Parameter &par, size_t nBeta, size_t nResponses, int method);

} // namespace lasso_regression
} // namespace algorithms
} // namespace daal

#endif
