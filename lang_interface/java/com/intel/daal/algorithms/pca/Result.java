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
 * @ingroup pca
 * @{
 */
package com.intel.daal.algorithms.pca;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__RESULT"></a>
 * @brief Provides methods to access final results obtained with the compute() method of PCA algorithm in the batch
 *        processing mode, or finalizeCompute() method in the online or distributed processing mode
 */
public class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result of the PCA algorithm
     * @param context   Context to manage the PCA algorithm result
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public Result(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns final result of the PCA algorithm
     * @param  id   Identifier of the result, @ref ResultId
     * @return Final result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.eigenValues && id != ResultId.eigenVectors && id != ResultId.means && id != ResultId.variances) {
            throw new IllegalArgumentException("id unsupported");
        }
        int idValue = id.getValue();
        return (NumericTable)Factory.instance().createObject(getContext(), cGetResultTable(cObject, idValue));
    }

    /**
     * Sets final result of the PCA algorithm
     * @param id    Identifier of the final result
     * @param value Object to store final result
     */
    public void set(ResultId id, NumericTable value) {
        if (id != ResultId.eigenValues && id != ResultId.eigenVectors && id != ResultId.means && id != ResultId.variances) {
            throw new IllegalArgumentException("id unsupported");
        }
        int idValue = id.getValue();
        cSetResultTable(cObject, idValue, value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultTable(long cResult, int id);

    private native void cSetResultTable(long cResult, int id, long cNumericTable);
}
/** @} */
