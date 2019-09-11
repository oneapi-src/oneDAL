/* file: ModelBuilder.java */
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

/**
 * @defgroup multi_class_classifier Multi-class Classifier
 * @brief Contains classes for computing the results of the multi-class classifier algorithm
 * @ingroup classification
 * @{
 */
package com.intel.daal.algorithms.multi_class_classifier;

import com.intel.daal.algorithms.multi_class_classifier.training.TrainingMethod;
import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTI_CLASS_CLASSIFIER__MODEL__BUILDER"></a>
 * @brief %Class for building model of the multi-class classifier algorithm
 *
 * @par References
 *      - Parameter class
 */
public class ModelBuilder extends SerializableBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingMethod method = TrainingMethod.oneAgainstOne;    /*!< %Computation method for the algorithm */

    /**
     * Constructs the multi class classifier model builder
     * @param context          Context to manage multi-class classifier model builder
     * @param method           Computation method for the algorithm
     * @param nFeatures        Number of features
     * @param nClasses         Number of classes
     */
    public ModelBuilder(DaalContext context, TrainingMethod method, long nFeatures, long nClasses) {
        super(context);
        this.method = method;
        this.cObject = cInit(this.method.getValue(), nFeatures, nClasses);
    }

    /**
     * Get built model of multi-class classifier
     * @return Model of multi-class classifier
     */
    public Model getModel() {
        return new Model(getContext(), cGetModel(this.cObject));
    }

    /**
     * Set two-class classifier model into a multi-class classifier model
     * @param negativeClassIdx Index of negative class for one vs one classification algorithm
     * @param positiveClassIdx Index of positive class for one vs one classification algorithm
     * @param model            Two-class classifier model to add into collection
     */
    public void setTwoClassClassifierModel(long negativeClassIdx, long positiveClassIdx, com.intel.daal.algorithms.classifier.Model model) {
        cSetTwoClassClassifierModel(this.cObject, negativeClassIdx, positiveClassIdx, model.getCObject());
    }

    private native long cInit(int method, long nFeatures, long nClasses);
    private native long cGetModel(long algAddr);
    private native void cSetTwoClassClassifierModel(long algAddr, long negativeClassIdx, long positiveClassIdx, long model);
}
/** @} */
