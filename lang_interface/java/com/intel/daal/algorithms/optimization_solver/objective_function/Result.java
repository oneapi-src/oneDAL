/* file: Result.java */
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
 * @ingroup objective_function
 * @{
 */
package com.intel.daal.algorithms.optimization_solver.objective_function;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULT"></a>
 * @brief Provides methods to access the results obtained with the compute() method of the
 *        Objective funtion algorithm in the batch processing mode
 */
public final class Result extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result for the Objective function algorithm
     * @param context       Context to manage the result for the Objective function algorithm
     */
    public Result(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    /**
     * Constructs the result for the Objective function algorithm
     * @param context       Context to manage the Objective function algorithm result
     * @param cResult       Pointer to C++ implementation of the result
     */
    public Result(DaalContext context, long cResult) {
        super(context, cResult);
    }

    /**
     * Returns the result of the Objective funtion algorithm in the batch processing mode
     * @param id Identifier of the result, @ref ResultId
     * @return Result that corresponds to the given identifier
     */
    public NumericTable get(ResultId id) {
        if (id != ResultId.gradientIdx && id != ResultId.valueIdx && id != ResultId.hessianIdx) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(),cGetResultNumericTable(cObject, id.getValue()));
    }

    /**
     * Sets the result of the Objective funtion algorithm in the batch processing mode
     * @param id Identifier of the result, @ref ResultId
     * @param val Object to store the result
     */
    public void set(ResultId id, NumericTable val) {
        if (id != ResultId.gradientIdx && id != ResultId.valueIdx && id != ResultId.hessianIdx) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultNumericTable(cObject, id.getValue(), val.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultNumericTable(long cObject, int id);

    private native void cSetResultNumericTable(long cResult, int id, long cNumericTable);
}
/** @} */
