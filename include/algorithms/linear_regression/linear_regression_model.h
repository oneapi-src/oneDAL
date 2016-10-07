/* file: linear_regression_model.h */
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
//  Implementation of the class defining the linear regression model
//--
*/

#ifndef __LINREG_MODEL_H__
#define __LINREG_MODEL_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "algorithms/model.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup linear_regression
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__PARAMETER"></a>
 * \brief Parameters for the linear regression algorithm
 *
 * \snippet linear_regression/linear_regression_model.h Parameter source code
 */
/* [Parameter source code] */
class DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
public:
    Parameter();
    bool interceptFlag; /*!< Flag that indicates whether the intercept needs to be computed */
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__MODEL"></a>
 * \brief %Base class for models trained with the linear regression algorithm
 *
 * \tparam modelFPType  Data type to store linear regression model data, double or float
 *
 * \par References
 *      - Parameter class
 *      - ModelNormEq class
 *      - ModelQR class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref training::interface1::Online "training::Online" class
 *      - \ref training::interface1::Distributed "training::Distributed" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::Model
{
public:
    /**
     * Constructs the linear regression model
     * \param[in] featnum Number of features in the training data
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Linear regression parameters
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    DAAL_EXPORT Model(size_t featnum, size_t nrhs, const Parameter &par, modelFPType dummy);

    /**
     * Constructs the linear regression model
     * \param[in] beta  Numeric table that contains linear regression coefficients
     * \param[in] par   Linear regression parameters
     */
    Model(data_management::NumericTablePtr &beta, const Parameter &par = Parameter());

    /**
     * Empty constructor for deserialization
     */
    Model() : daal::algorithms::Model()
    {}

    /**
     * Initializes linear regression coefficients of the linear regression model
     */
    virtual void initialize();

    virtual ~Model()
    {}

    /**
     * Returns the number of regression coefficients
     * \return Number of regression coefficients
     */
    size_t getNumberOfBetas();

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    size_t getNumberOfFeatures();

    /**
     * Returns the number of responses in the training data set
     * \return Number of responses in the training data set
     */
    size_t getNumberOfResponses();

    /**
     * Returns true if the linear regression model contains the intercept term, and false otherwise
     * \return True if the linear regression model contains the intercept term, and false otherwise
     */
    bool getInterceptFlag();

    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    data_management::NumericTablePtr getBeta();

    /**
     * Returns the serialization tag of the linear regression model
     * \return         Serialization tag of the linear regression model
     */

    int getSerializationTag() DAAL_C11_OVERRIDE  { return 0; }
    /**
     *  Serializes a linear regression model object
     *  \param[in]  archive  Storage for a serialized model object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE {}

    /**
     *  Deserializes a linear regression model object
     *  \param[in]  archive  Storage for a deserialized model object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE {}

protected:
    bool _interceptFlag;     /* Flag. True if the linear regression model contains the intercept term;
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

/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Model;

/**
 * Checks the correctness of linear regression model
 * \param[in]  model             The model to check
 * \param[in]  par               The parameter of linear regression algorithm
 * \param[out] errors            The collection of errors
 * \param[in]  coefdim           Required number of linear regression coefficients
 * \param[in]  nrhs              Required number of responses on the training stage
 * \param[in]  method            Computation method
 */
DAAL_EXPORT void checkModel(linear_regression::Model* model, const daal::algorithms::Parameter *par, services::ErrorCollection *errors,
    const size_t coefdim, const size_t nrhs, int method);

}
}
}
#endif
