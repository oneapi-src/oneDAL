/* file: PartialModel.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__PARTIALMODEL"></a>
 * @brief Multinomial naive Bayes PartialModel
 */
public class PartialModel extends com.intel.daal.algorithms.classifier.Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public PartialModel(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Returns Numeric Table containing group sums with class label used as an index
     *  @return Numeric Table containing group sums with class label used as an index
     */
    public NumericTable getClassGroupSum() {
        return new HomogenNumericTable(getContext(), cGetClassGroupSum(this.getCObject()));
    }

    /**
     * Returns Numeric Table containing sum of all observations and features for each class
     *  @return Numeric Table containing sum of all observations and features for each class
     */
    public NumericTable getClassSize() {
        return new HomogenNumericTable(getContext(), cGetClassSize(this.getCObject()));
    }

    /**
     * Returns number of features in training dataset
     *  @return Number of features in training dataset
     */
    public long getNFeatures() {
        return cGetNFeatures(this.getCObject());
    }

    /**
     * Returns number of observations in training dataset
     *  @return Number of observations in training dataset
     */
    public long getNObservations() {
        return cGetNObservations(this.getCObject());
    }

    private native long cGetClassGroupSum(long modelAddr);

    private native long cGetClassSize(long modelAddr);

    private native long cGetNFeatures(long modelAddr);

    private native long cGetNObservations(long modelAddr);
}
