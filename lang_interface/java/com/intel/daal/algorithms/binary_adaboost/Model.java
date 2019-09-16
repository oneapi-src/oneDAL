/* file: Model.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/**
 * @ingroup binary_adaboost
 */
package com.intel.daal.algorithms.binary_adaboost;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__binary_adaboost__MODEL"></a>
 * @brief %Model of the classifier trained by the binary_adaboost algorithm in the batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.boosting.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Returns the numeric table that contains the array of weights of weak learners constructed
     * during training of the binary_adaboost algorithm.
     * The size of the array equals the number of weak learners
     * @return Array of weights of weak learners.
     */
    public NumericTable getAlpha() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetAlpha(this.getCObject()));
    }

    private native long cGetAlpha(long modelAddr);
}
/** @} */
