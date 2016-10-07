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

package com.intel.daal.algorithms.brownboost;

import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__MODEL"></a>
 * @brief %Model of the classifier trained by the BrownBoost algorithm in the batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.boosting.Model {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Returns the numeric table that contains the array of weights of weak learners constructed
     * during training of the BrownBoost algorithm.
     * The size of the array equals the number of weak learners
     * @return Array of weights of weak learners.
     */
    public NumericTable getAlpha() {
        return new HomogenNumericTable(getContext(), cGetAlpha(this.getCObject()));
    }

    private NumericTable _alpha;

    private native long cGetAlpha(long modelAddr);
}
