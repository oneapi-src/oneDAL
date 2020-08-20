/* file: svm_classification_model.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef __SVM_CLASSIFICATION_MODEL_H__
#define __SVM_CLASSIFICATION_MODEL_H__

#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/kernel_function/kernel_function_linear.h"
#include "algorithms/kernel_function/kernel_function_types.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/svm/svm_parameter.h"

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
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL"></a>
 * \brief %Model of the classifier trained by the svm::training::Batch algorithm
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public daal::algorithms::classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model);

    // Added create() methods into models and numeric tables clasees. Reviewer: Natalia Shakhalova. Approver: Ivan Kuzmin. 2017.07.17

    /**
     * Constructs the SVM model
     * \tparam modelFPType  Data type to store SVM model data, double or float
     * \param[in] nColumns  Number of features in input data
     * \param[in] layout    Data layout of the numeric table of support vectors
     * \param[out] stat     Status of the model construction
     * \return SVM model
     */
    template <typename modelFPType>
    DAAL_DEPRECATED DAAL_EXPORT static services::SharedPtr<Model> create(
        size_t nColumns, data_management::NumericTableIface::StorageLayout layout = data_management::NumericTableIface::aos,
        services::Status * stat = NULL);

    /**
     * Constructs empty SVM model for deserialization
     * \param[out] stat     Status of the model construction
     * \return Empty SVM model for deserialization
     */
    DAAL_DEPRECATED static services::SharedPtr<Model> create(services::Status * stat = NULL)
    {
        services::SharedPtr<Model> modelPtr(new Model());
        if (!modelPtr)
        {
            if (stat) stat->add(services::ErrorMemoryAllocationFailed);
        }
        return modelPtr;
    }

    virtual ~Model() {}

    /**
     * Returns support vectors constructed during the training of the SVM model
     * \return Array of support vectors
     */
    DAAL_DEPRECATED data_management::NumericTablePtr getSupportVectors() { return _SV; }

    /**
     * Returns indices of the support vectors constructed during the training of the SVM model
     * \return Array of support vectors indices
     */
    DAAL_DEPRECATED data_management::NumericTablePtr getSupportIndices() { return _SVIndices; }

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * \return Array of classification coefficients
     */
    DAAL_DEPRECATED data_management::NumericTablePtr getClassificationCoefficients() { return _SVCoeff; }

    /**
     * Returns the bias constructed during the training of the SVM model
     * \return Bias
     */
    virtual double getBias() { return _bias; }

    /**
     * Sets the bias for the SVM model
     * \param bias  Bias of the model
     */
    virtual void setBias(double bias) { _bias = bias; }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    DAAL_DEPRECATED size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE { return (_SV ? _SV->getNumberOfColumns() : 0); }

    Model();

protected:
    data_management::NumericTablePtr _SV;        /*!< \private Support vectors */
    data_management::NumericTablePtr _SVCoeff;   /*!< \private Classification coefficients */
    double _bias;                                /*!< \private Bias of the distance function D(x) = w*Phi(x) + bias */
    data_management::NumericTablePtr _SVIndices; /*!< \private Indices of the support vectors in training data set */

    template <typename modelFPType>
    DAAL_EXPORT Model(modelFPType dummy, size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status & st);

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status st = classifier::Model::serialImpl<Archive, onDeserialize>(arch);
        if (!st) return st;
        arch->setSharedPtrObj(_SV);
        arch->setSharedPtrObj(_SVCoeff);
        arch->set(_bias);

        arch->setSharedPtrObj(_SVIndices);

        return st;
    }
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1

using interface1::Model;
using interface1::ModelPtr;

namespace classification
{
namespace interface1
{
/**
 * @ingroup svm
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__SVM__CLASSIFICATION__PARAMETER"></a>
 * \brief Optional parameters
 *
 * \snippet svm/svm_classification_model.h Parameter source code
/* [interface1::Parameter source code] */

struct DAAL_EXPORT Parameter : public classifier::Parameter, public daal::algorithms::svm::Parameter
{
    Parameter(const services::SharedPtr<kernel_function::KernelIface> & kernelForParameter =
                  services::SharedPtr<kernel_function::KernelIface>(new kernel_function::linear::Batch<>()))
        : daal::algorithms::svm::Parameter(kernelForParameter)
    {}

    services::Status check() const DAAL_C11_OVERRIDE;
};
/* [interface1::Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL"></a>
 * \brief %Model of the classifier trained by the svm::training::Batch algorithm
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public classifier::Model
{
public:
    DECLARE_MODEL(Model, classifier::Model);

    virtual ~Model() {}

    class ModelImpl;
    typedef services::SharedPtr<ModelImpl> ModelImplPtr;

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    const ModelImpl * impl() const { return _impl.get(); }

    /**
     * Returns actual model implementation
     * \return Model implementation
     */
    ModelImpl * impl() { return _impl.get(); }

    /**
     * Constructs the SVM model
     * \tparam modelFPType  Data type to store SVM model data, double or float
     * \param[in] nColumns  Number of features in input data
     * \param[in] layout    Data layout of the numeric table of support vectors
     * \param[out] stat     Status of the model construction
     * \return SVM model
     */
    template <typename modelFPType>
    DAAL_EXPORT static services::SharedPtr<Model> create(
        size_t nColumns, data_management::NumericTableIface::StorageLayout layout = data_management::NumericTableIface::aos,
        services::Status * stat = NULL);

    /**
     * Constructs empty SVM model for deserialization
     * \param[out] stat     Status of the model construction
     * \return Empty SVM model for deserialization
     */
    static services::SharedPtr<Model> create(services::Status * stat = NULL);

    /**
     * Returns support vectors constructed during the training of the SVM model
     * \return Array of support vectors
     */
    data_management::NumericTablePtr getSupportVectors();

    /**
     * Returns indices of the support vectors constructed during the training of the SVM model
     * \return Array of support vectors indices
     */
    data_management::NumericTablePtr getSupportIndices();
    /**
     * Returns coefficients constructed during the training of the SVM model
     * \return Array of classification coefficients
     */
    data_management::NumericTablePtr getCoefficients();

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * \return Array of classification coefficients
     */
    DAAL_DEPRECATED data_management::NumericTablePtr getClassificationCoefficients() { return getCoefficients(); }

    /**
     * Returns the bias constructed during the training of the SVM model
     * \return Bias
     */
    double getBias();

    /**
     * Sets the bias for the SVM model
     * \param bias  Bias of the model
     */
    void setBias(double bias);

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

protected:
    template <typename modelFPType>
    DAAL_EXPORT Model(modelFPType dummy, size_t nColumns, data_management::NumericTableIface::StorageLayout layout, services::Status & st);

    // services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;

    // services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

private:
    ModelImplPtr _impl; /*!< Model implementation */
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1

using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace classification
} // namespace svm
/** @} */
} // namespace algorithms
} // namespace daal
#endif
