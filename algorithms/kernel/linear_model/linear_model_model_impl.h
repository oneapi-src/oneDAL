/* file: linear_model_model_impl.h */
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
//  Declaration of the class that implements the linear model
//--
*/

#ifndef __LINEAR_MODEL_MODEL_IMPL_H__
#define __LINEAR_MODEL_MODEL_IMPL_H__

#include "linear_model_model.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace internal
{
using namespace data_management;

/**
 * \brief Class that implements the methods for models trained with the linear model algorithm
 */
class ModelInternal
{
public:
    /**
     * Constructs the linear model
     * \param[in] nFeatures  Number of features in the training data
     * \param[in] nResponses Number of responses in the training data
     * \param[in] par        Parameters of the linear model
     * \param[in] dummy      Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    ModelInternal(size_t nFeatures, size_t nResponses, const linear_model::Parameter &par, modelFPType dummy);

    /**
     * Constructs the linear model
     * \param[in] beta  Numeric table that contains the linear model coefficients
     * \param[in] par   Parameters of the linear model
     */
    ModelInternal(const data_management::NumericTablePtr &beta, const linear_model::Parameter &par = Parameter());

    ModelInternal();

    virtual ~ModelInternal()
    {}

    /**
     * Initializes the coefficients of the linear model
     */
    services::Status initialize();

    /**
     * Returns the number of regression coefficients
     * \return Number of regression coefficients
     */
    size_t getNumberOfBetas() const;

    /**
     * Returns the number of responses in the training data set
     * \return Number of responses in the training data set
     */
    size_t getNumberOfResponses() const;

    /**
     * Returns true if the regression model contains the intercept term, and false otherwise
     * \return True if the regression model contains the intercept term, and false otherwise
     */
    bool getInterceptFlag() const;

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    size_t getNumberOfFeatures() const;
    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    data_management::NumericTablePtr getBeta();

protected:
    bool _interceptFlag;     /* Flag. True if the ridge regression model contains the intercept term;
                                false otherwise. */
    data_management::NumericTablePtr _beta;    /* Table that contains resulting coefficients */

    services::Status setToZero(data_management::NumericTable &table);

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        arch->set(_interceptFlag);

        arch->setSharedPtrObj(_beta);

        return services::Status();
    }
};

/**
 * \brief Class that connects the interface and implementation
 */
class ModelImpl : public linear_model::Model,
                  public ModelInternal
{
public:
    typedef ModelInternal   ImplType;

    /**
     * Constructs the linear model
     * \param[in] nFeatures  Number of features in the training data
     * \param[in] nResponses Number of responses in the training data
     * \param[in] par        Parameters of the linear model
     * \param[in] dummy      Dummy variable for the templated constructor
     */
    template<typename modelFPType>
    ModelImpl(size_t nFeatures, size_t nResponses, const linear_model::Parameter &par, modelFPType dummy) :
        ImplType(nFeatures, nResponses, par, dummy)
    {}

    /**
     * Constructs the linear model
     * \param[in] beta  Numeric table that contains the linear model coefficients
     * \param[in] par   Parameters of the linear model
     */
    ModelImpl(const data_management::NumericTablePtr &beta, const linear_model::Parameter &par = Parameter()) :
        ImplType(beta, par)
    {}

    /**
     * Initializes the coefficients of the linear model
     */
    services::Status initialize() DAAL_C11_OVERRIDE { return ImplType::initialize(); }

    /**
     * Returns the number of regression coefficients
     * \return Number of regression coefficients
     */
    size_t getNumberOfBetas() const DAAL_C11_OVERRIDE  { return ImplType::getNumberOfBetas(); }

    /**
     * Returns the number of responses in the training data set
     * \return Number of responses in the training data set
     */
    size_t getNumberOfResponses() const DAAL_C11_OVERRIDE  { return ImplType::getNumberOfResponses(); }

    /**
     * Returns true if the regression model contains the intercept term, and false otherwise
     * \return True if the regression model contains the intercept term, and false otherwise
     */
    bool getInterceptFlag() const DAAL_C11_OVERRIDE { return ImplType::getInterceptFlag(); }

    /**
     * Returns the number of features in the training data set
     * \return Number of features in the training data set
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return ImplType::getNumberOfFeatures(); }

    /**
     * Returns the numeric table that contains regression coefficients
     * \return Table that contains regression coefficients
     */
    data_management::NumericTablePtr getBeta() DAAL_C11_OVERRIDE { return ImplType::getBeta(); }
};

}
}
}
}
#endif
