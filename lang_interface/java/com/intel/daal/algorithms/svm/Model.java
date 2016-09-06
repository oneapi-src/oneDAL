/* file: Model.java */
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

package com.intel.daal.algorithms.svm;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
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
        System.loadLibrary("JavaAPI");
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
        return new HomogenNumericTable(getContext(), cGetSupportVectors(this.getCObject()));
    }

    /**
     * Returns classification coefficients constructed during the training of the SVM model
     * @return Array of classification coefficients
     */
    public NumericTable getClassificationCoefficients() {
        return new HomogenNumericTable(getContext(), cGetClassificationCoefficients(this.getCObject()));
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
