/* file: svm_model.h */
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
//  Implementation of the class defining the SVM model.
//--
*/

#ifndef __SVM_MODEL_H__
#define __SVM_MODEL_H__

#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/kernel_function/kernel_function_linear.h"
#include "algorithms/kernel_function/kernel_function_types.h"
#include "algorithms/classifier/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup svm Support Vector Machine Classifier
 * \copydoc daal::algorithms::svm
 * @ingroup classification
 * @{
 */
/**
 * \brief Contains classes to work with the support vector machine classifier
 */
namespace svm
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @ingroup svm
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__SVM__PARAMETER"></a>
 * \brief Optional parameters
 *
 * \ref opt_notice
 *
 * \snippet svm/svm_model.h Parameter source code
 */
/* [Parameter source code] */
struct Parameter : public classifier::Parameter
{
    Parameter(const services::SharedPtr<kernel_function::KernelIface> &kernel
              = services::SharedPtr<kernel_function::KernelIface>(new kernel_function::linear::Batch<>()),
              double C = 1.0,
              double accuracyThreshold = 0.001,
              double tau = 1.0e-6,
              size_t maxIterations = 1000000,
              size_t cacheSize = 8000000,
              bool doShrinking = true,
              size_t shrinkingStep = 1000) :
        C(C), accuracyThreshold(accuracyThreshold), tau(tau), maxIterations(maxIterations), cacheSize(cacheSize),
        doShrinking(doShrinking), shrinkingStep(shrinkingStep), kernel(kernel) {};

    double C;                   /*!< Upper bound in constraints of the quadratic optimization problem */
    double accuracyThreshold;   /*!< Training accuracy */
    double tau;                 /*!< Tau parameter of the working set selection scheme */
    size_t maxIterations;       /*!< Maximal number of iterations for the algorithm */
    size_t cacheSize;           /*!< Size of cache in bytes to store values of the kernel matrix.
                                     A non-zero value enables use of a cache optimization technique */
    bool doShrinking;           /*!< Flag that enables use of the shrinking optimization technique */
    size_t shrinkingStep;       /*!< Number of iterations between the steps of shrinking optimization technique */
    services::SharedPtr<kernel_function::KernelIface> kernel;   /*!< Kernel function */

    void check() const DAAL_C11_OVERRIDE
    {
        classifier::Parameter::check();
        if(C <= 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, cBoundStr()));
            return;
        }
        if(accuracyThreshold <= 0 || accuracyThreshold >= 1)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, accuracyThresholdStr()));
            return;
        }
        if(tau <= 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, tauStr()));
            return;
        }
        if(maxIterations == 0)
        {
            this->_errors->add(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
            return;
        }
        if(!kernel.get())
        {
            this->_errors->add(services::Error::create(services::ErrorNullAuxiliaryAlgorithm, services::ParameterName, kernelFunctionStr()));
            return;
        }
    }
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL"></a>
 * \brief %Model of the classifier trained by the svm::training::Batch algorithm
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class Model : public classifier::Model
{
public:
    DAAL_DOWN_CAST_OPERATOR(Model,classifier::Model)

    /**
     * Constructs the SVM model
     * \tparam modelFPType  Data type to store SVM model data, double or float
     * \param[in] dummy     Dummy variable for the templated constructor
     * \param[in] layout    Data layout of the numeric table of support vectors
     */
    template<typename modelFPType>
    Model(modelFPType dummy, data_management::NumericTableIface::StorageLayout layout = data_management::NumericTableIface::aos) :
            classifier::Model(), _bias(0.0)
    {
        if (layout == data_management::NumericTableIface::csrArray)
        {
            modelFPType *dummyPtr = NULL;
            _SV = data_management::NumericTablePtr(new data_management::CSRNumericTable(dummyPtr));
        }
        else
        {
            _SV = data_management::NumericTablePtr(new data_management::HomogenNumericTable<modelFPType>(NULL, 0));
        }
        _SVCoeff = data_management::NumericTablePtr(new data_management::HomogenNumericTable<modelFPType>(NULL, 1));
    }

    /**
     * Empty constructor for deserialization
     */
    Model() : _SV(), _SVCoeff(), _bias(0.0) {}

    /* Destructor */
    virtual ~Model() {}

    /**
     * Returns support vectors constructed during the training of the SVM model
     * \return Array of support vectors
     */
    data_management::NumericTablePtr getSupportVectors() { return _SV; }

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * \return Array of classification coefficients
     */
    data_management::NumericTablePtr getClassificationCoefficients() { return _SVCoeff; }

    /**
     * Returns the bias constructed during the training of the SVM model
     * \return Bias
     */
    virtual double getBias() { return _bias; }

    /**
     * Sets the bias for the SVM model
     * \param bias  Bias of the model
     */
    virtual void setBias(double bias)
    {
        _bias = bias;
    }

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_SVM_MODEL_ID; }
    /**
     *  Serializes the model object
     *  \param[in]  archive  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(archive);}

    /**
     *  Deserializes the model object
     *  \param[in]  archive  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *archive) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(archive);}

protected:
    data_management::NumericTablePtr _SV;          /*!< \private Support vectors */
    data_management::NumericTablePtr _SVCoeff;     /*!< \private Classification coefficients */
    double _bias;                         /*!< \private Bias of the distance function D(x) = w*Phi(x) + bias */

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        arch->setSharedPtrObj(_SV);
        arch->setSharedPtrObj(_SVCoeff);
        arch->set(_bias);
    }
};
/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Model;

} // namespace svm
/** @} */
} // namespace algorithms
} // namespace daal
#endif
