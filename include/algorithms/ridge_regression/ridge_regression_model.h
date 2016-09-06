/* file: ridge_regression_model.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of the class defining the ridge regression model
//--
*/

#ifndef __RIDGE_REGRESSION_MODEL_H__
#define __RIDGE_REGRESSION_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup ridge_regression Ridge Regression
 * \copydoc daal::algorithms::ridge_regression
 * @ingroup regression
 * @{
 */
namespace ridge_regression
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__RIDGE_REGRESSION__PARAMETER"></a>
 * \brief Parameters for the ridge regression algorithm
 *
 * \snippet ridge_regression/ridge_regression_model.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{

    Parameter();

    bool interceptFlag; /*!< Flag that indicates whether the intercept needs to be computed */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__RIDGE_REGRESSION__TRAINPARAMETER"></a>
 * \brief Parameters for the ridge regression algorithm
 *
 * \snippet ridge_regression/ridge_regression_model.h TrainParameter source code
 */
/* [TrainParameter source code] */
struct DAAL_EXPORT TrainParameter : public Parameter
{
    TrainParameter();

    void check() const DAAL_C11_OVERRIDE;

    data_management::NumericTablePtr ridgeParameters; /*!< Numeric table that contains values of ridge parameters */
};
/* [TrainParameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__RIDGE_REGRESSION__MODEL"></a>
 * \brief %Base class for models trained with the ridge regression algorithm
 *
 * \tparam modelFPType  Data type to store ridge regression model data, double or float
 *
 * \par References
 *      - Parameter class
 *      - ModelNormEq class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::Model
{
public:
    /**
     * Constructs the ridge regression model
     * \param[in] featnum Number of features in the training data
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Ridge regression parameters
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    DAAL_EXPORT Model(size_t featnum, size_t nrhs, const Parameter &par, modelFPType dummy);

    /**
     * Constructs the ridge regression model
     * \param[in] beta  Numeric table that contains ridge regression coefficients
     * \param[in] par   ridge regression parameters
     */
    Model(data_management::NumericTablePtr &beta, const Parameter &par = Parameter());

    /**
     * Empty constructor for deserialization
     */
    Model() : daal::algorithms::Model() {}

    /**
     * Initializes ridge regression coefficients of the ridge regression model
     */
    virtual void initialize();

    virtual ~Model() {}

    /**
     * Returns the number of regression coefficients
     * \return Number of regression coefficients
     */
    size_t getNumberOfBetas() const;

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    size_t getNumberOfFeatures() const;

    /**
     * Returns the number of responses in the training data set
     * \return Number of responses in the training data set
     */
    size_t getNumberOfResponses() const;

    /**
     * Returns true if the ridge regression model contains the intercept term, and false otherwise
     * \return True if the ridge regression model contains the intercept term, and false otherwise
     */
    bool getInterceptFlag() const;

    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    data_management::NumericTablePtr getBeta();

    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    data_management::NumericTablePtr getBeta() const;

protected:
    bool _interceptFlag;     /* Flag. True if the ridge regression model contains the intercept term;
                                 false otherwise. */
    size_t _coefdim;        /* Number of regression coefficients */
    size_t _nrhs;           /* Number of responses in the training data set */
    data_management::NumericTablePtr _beta;    /* Table that contains resulting coefficients */

    void setToZero(data_management::NumericTable *table);

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::Model::serialImpl<Archive, onDeserialize>(arch);

        arch->set(_interceptFlag);
        arch->set(_coefdim      );
        arch->set(_nrhs         );

        arch->setSharedPtrObj(_beta);
    }
};

} // namespace interface1

using interface1::Parameter;
using interface1::TrainParameter;
using interface1::Model;

/**
 * Checks the correctness of ridge regression model
 * \param[in]  model             The model to check
 * \param[in]  par               The parameter of ridge regression algorithm
 * \param[out] errors            The collection of errors
 * \param[in]  coefdim           Required number of ridge regression coefficients
 * \param[in]  nrhs              Required number of responses on the training stage
 * \param[in]  method            Computation method
 */
DAAL_EXPORT void checkModel(ridge_regression::Model* model, const daal::algorithms::Parameter *par, services::ErrorCollection *errors,
    const size_t coefdim, const size_t nrhs, int method);

} // namespace ridge_regression
/** @} */
} // namespace algorithms
} // namespace daal

#endif
