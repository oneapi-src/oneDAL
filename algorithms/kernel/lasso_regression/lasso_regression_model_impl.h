/* file: lasso_regression_model_impl.h */
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

#ifndef __LASSO_REGRESSION_MODEL_IMPL__
#define __LASSO_REGRESSION_MODEL_IMPL__

#include "algorithms/lasso_regression/lasso_regression_model.h"
#include "algorithms/lasso_regression/lasso_regression_training_types.h"

#include "linear_model_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{
namespace internal
{

class ModelImpl : public lasso_regression::Model, public linear_model::internal::ModelInternal
{
public:
    typedef linear_model::internal::ModelInternal ImplType;

    /**
     * Constructs the lasso regression model for the normal equations method
     * \param[in] featnum Number of features in the training data set
     * \param[in] nrhs    Number of responses in the training data
     * \param[in] par     Parameters of lasso regression model-based training
     * \param[in] dummy   Dummy variable for the templated constructor
     */
    template <typename modelFPType>
    ModelImpl(size_t featnum, size_t nrhs, const lasso_regression::training::Parameter &par, modelFPType dummy, services::Status &s) :
        ImplType(featnum, nrhs, par, dummy)
    {}

    ModelImpl() {}

    virtual ~ModelImpl() {}

    /**
     * Initializes the lasso regression model
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

protected:

    services::Status serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        ImplType::serialImpl<data_management::InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        ImplType::serialImpl<const data_management::OutputDataArchive, true>(arch);

        return services::Status();
    }

};

} // namespace internal
} // namespace lasso_regression
} // namespace algorithms
} // namespace daal

#endif
