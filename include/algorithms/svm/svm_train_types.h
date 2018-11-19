/* file: svm_train_types.h */
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
//  SVM parameter structure
//--
*/

#ifndef __SVM_TRAIN_TYPES_H__
#define __SVM_TRAIN_TYPES_H__

#include "algorithms/algorithm.h"
#include "algorithms/kernel_function/kernel_function.h"
#include "algorithms/svm/svm_model.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace svm
{
/**
 * @defgroup svm_training Training
 * \copydoc daal::algorithms::svm::training
 * @ingroup svm
 * @{
 */
/**
 * \brief Contains classes to train the SVM model
 */
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__SVM__TRAINING__METHOD"></a>
 * Available methods to train the SVM model
 */
enum Method
{
    boser = 0,          /*!< Method proposed by Boser et al. */
    defaultDense = 0    /*!< Default method */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the
 *        SVM training algorithm in the batch processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();

    virtual ~Result() {}

    /**
     * Returns the model trained with the SVM algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the SVM algorithm
     */
    daal::algorithms::svm::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Allocates memory for storing SVM training results
     * \param[in] input     Pointer to input structure
     * \param[in] parameter Pointer to parameter structure
     * \param[in] method    Algorithm method
     *
     * \return Status of computation
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace svm
} // namespace algorithms
} // namespace daal
#endif
