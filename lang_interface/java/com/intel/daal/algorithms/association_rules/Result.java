/* file: Result.java */
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

/**
 * @ingroup association_rules
 * @{
 */
package com.intel.daal.algorithms.association_rules;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__RESULT"></a>
 * @brief Results obtained with the compute() method of the association rules algorithm in the batch processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result
     * @param context  Context to manage the result
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cObject) {
        super(context, cObject);
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
        return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, id.getValue()));
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

    private native long cGetResultTable(long resAddr, int id);

    private native void cSetResultTable(long resAddr, int id, long ntAddr);

}
/** @} */
