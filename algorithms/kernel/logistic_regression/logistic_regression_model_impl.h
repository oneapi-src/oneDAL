/* file: logistic_regression_model_impl.h */
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
//  Implementation of the class defining the logistic regression model
//--
*/

#ifndef __LOGISTIC_REGRESSION_MODEL_IMPL__
#define __LOGISTIC_REGRESSION_MODEL_IMPL__

#include "algorithms/logistic_regression/logistic_regression_model.h"
#include "../classifier/classifier_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace logistic_regression
{
namespace internal
{

class ModelImpl : public logistic_regression::Model, public algorithms::classifier::internal::ModelInternal
{
public:
    typedef algorithms::classifier::internal::ModelInternal ClassificationImplType;

    ModelImpl(size_t nFeatures = 0, bool interceptFlag = true);
    template <typename modelFPType>
    ModelImpl(size_t nFeatures, bool interceptFlag, size_t nClasses, modelFPType dummy, services::Status* st);
    ~ModelImpl(){}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE{ return ClassificationImplType::getNumberOfFeatures(); }

    //Implementation of classification::Model
    virtual size_t getNumberOfBetas() const DAAL_C11_OVERRIDE;
    virtual bool getInterceptFlag() const DAAL_C11_OVERRIDE;
    virtual data_management::NumericTablePtr getBeta() DAAL_C11_OVERRIDE;
    virtual const data_management::NumericTablePtr getBeta() const DAAL_C11_OVERRIDE;

    virtual services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;
    virtual services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

    services::Status reset(bool interceptFlag);
    static logistic_regression::ModelPtr create(size_t nFeatures, bool interceptFlag, services::Status *stat = nullptr);

protected:
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->set(ClassificationImplType::_nFeatures);
        arch->set(_interceptFlag);
        arch->setSharedPtrObj(_beta);
        return services::Status();
    }

protected:
    bool _interceptFlag; // True if the model contains the intercept term false otherwise
    data_management::NumericTablePtr _beta;  // Model coefficients
};

} // namespace internal
} // namespace logistic_regression
} // namespace algorithms
} // namespace daal

#endif
