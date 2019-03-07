/* file: Model.java */
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
 * @defgroup svm Support Vector Machine Classifier
 * @brief Contains classes to work with the support vector machine classifier
 * @ingroup classification
 * @{
 */
package com.intel.daal.algorithms.svm;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVM__MODEL"></a>
 * @brief %Model of the classifier trained by the svm.training.TrainingBatch algorithm.
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    private Precision                 prec; /*!< Precision of intermediate computations */
    private double    _bias; /*!< Bias of the model */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
        _bias = cGetBias(cModel);
    }

    /**
     * Returns support vectors constructed during the training of the SVM model
     * @return Array of support vectors
     */
    public NumericTable getSupportVectors() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetSupportVectors(this.getCObject()));
    }

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * @return Array of classification coefficients
     */
    public NumericTable getClassificationCoefficients() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetClassificationCoefficients(this.getCObject()));
    }

    /**
     * Returns the bias constructed during the training of the SVM model
     * @return Bias
     */
    public double getBias() {
        return _bias;
    }

    private native long cGetSupportVectors(long modelAddr);

    private native long cGetClassificationCoefficients(long modelAddr);

    private native double cGetBias(long modelAddr);
}
/** @} */
