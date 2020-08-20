/* file: svm_regression_model.h */
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

#ifndef __SVM_regression_MODEL_H__
#define __SVM_regression_MODEL_H__

#include "data_management/data/homogen_numeric_table.h"
#include "data_management/data/csr_numeric_table.h"
#include "algorithms/model.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/kernel_function/kernel_function_linear.h"
#include "algorithms/kernel_function/kernel_function_types.h"
#include "algorithms/regression/classifier_model.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup svm Support Vector Machine Classifier
 * \copydoc daal::algorithms::svm
 * @ingroup regression
 * @{
 */
/**
 * \brief Contains classes to work with the support vector machine regression
 */
namespace svm
{
namespace regression
{
namespace interface1
{
/**
 * @ingroup svm
 * @{
 */
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__SVM__REGRESSION__PARAMETER"></a>
 * \brief Optional parameters
 *
 * \snippet svm/svm_regression_model.h Parameter source code
/* [interface1::Parameter source code] */

struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter, public daal::algorithms::svm::Parameter
{
    Parameter(const services::SharedPtr<kernel_function::KernelIface> & kernelForParameter =
                  services::SharedPtr<kernel_function::KernelIface>(new kernel_function::linear::Batch<>()))
        : daal::algorithms::svm::Parameter(kernelForParameter), epsilon(0.1)
    {}

    services::Status check() const DAAL_C11_OVERRIDE;

    double epsilon; /*!< The error tolerance parameter of the loss function for regression task. */
};
/* [interface1::Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL"></a>
 * \brief %Model of the regression trained by the svm::training::Batch algorithm
 *
 * \par References
 *      - Parameter class
 *      - \ref training::interface1::Batch "training::Batch" class
 *      - \ref prediction::interface1::Batch "prediction::Batch" class
 */
class DAAL_EXPORT Model : public regression::Model
{
public:
    DECLARE_MODEL(Model, regression::Model);

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
     * \return Array of regression coefficients
     */
    data_management::NumericTablePtr getCoefficients();

    /**
     * Returns the bias constructed during the training of the SVM model
     * \return Bias
     */
    virtual double getBias();

    /**
     * Sets the bias for the SVM model
     * \param bias  Bias of the model
     */
    virtual void setBias(double bias);

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

protected:
    services::Status serializeImpl(data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE;

    services::Status deserializeImpl(const data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE;

private:
    ModelImplPtr _impl; /*!< Model implementation */
};
typedef services::SharedPtr<Model> ModelPtr;
/** @} */
} // namespace interface1

using interface1::Parameter;
using interface1::Model;
using interface1::ModelPtr;

} // namespace regression
} // namespace svm
/** @} */
} // namespace algorithms
} // namespace daal
#endif
