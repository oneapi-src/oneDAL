/* file: Result.java */
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

package com.intel.daal.algorithms.association_rules;

import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__RESULT"></a>
 * @brief Results obtained with the compute() method of the association rules algorithm in the batch processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cAlgorithm, Precision prec, Method method, ComputeMode cmode) {
        super(context);
        cObject = cGetResult(cAlgorithm, prec.getValue(), method.getValue(), cmode.getValue());
    }

    /**
     * Returns the final result of the association rules algorithm
     * @param id   Identifier of the result
     * @return Final result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.largeItemsets && id != ResultId.largeItemsetsSupport && id != ResultId.antecedentItemsets
                && id != ResultId.consequentItemsets && id != ResultId.confidence) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new HomogenNumericTable(getContext(), cGetResultTable(cObject, id.getValue()));
    }

    /**
     * Sets the final result of the association rules algorithm
     * @param id   Identifier of the result
     * @param val  Object to store the final result
     */
    public void set(ResultId id, NumericTable val) {
        if (id != ResultId.largeItemsets && id != ResultId.largeItemsetsSupport && id != ResultId.antecedentItemsets
                && id != ResultId.consequentItemsets && id != ResultId.confidence) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultTable(cObject, id.getValue(), val.getCObject());
    }

    private native long cNewResult();

    private native long cGetResult(long cAlgorithm, int prec, int method, int mode);

    private native long cGetResultTable(long resAddr, int id);

    private native void cSetResultTable(long resAddr, int id, long ntAddr);

}
