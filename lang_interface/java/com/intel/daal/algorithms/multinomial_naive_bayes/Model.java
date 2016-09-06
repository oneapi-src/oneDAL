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

package com.intel.daal.algorithms.multinomial_naive_bayes;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__MODEL"></a>
 * @brief Model of the multinomial naive Bayes classifier trained in the batch processing mode
 */
public class Model extends com.intel.daal.algorithms.classifier.Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Returns Numeric Table with logarithms of prior probabilities of classes
     *  @return Numeric Table with logarithms of prior probabilities of classes
     */
    public NumericTable getLogP() {
        return new HomogenNumericTable(getContext(), cGetLogP(this.getCObject()));
    }

    /**
     * Returns Numeric Table with logarithms of the conditional probabilities of features given a class
     *  @return Numeric Table with logarithms of the conditional probabilities of features given a class
     */
    public NumericTable getLogTheta() {
        return new HomogenNumericTable(getContext(), cGetLogTheta(this.getCObject()));
    }

    /**
     * Returns number of features in training dataset
     *  @return Number of features in training dataset
     */
    public long getNFeatures() {
        return cGetNFeatures(this.getCObject());
    }

    private native long cGetLogP(long modelAddr);

    private native long cGetLogTheta(long modelAddr);

    private native long cGetNFeatures(long modelAddr);
}
