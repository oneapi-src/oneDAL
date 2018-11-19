/* file: DistributedStep4LocalInput.java */
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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDSTEP4LOCALINPUT"></a>
 * @brief %Input objects for the implicit ALS training algorithm in the fourth step of the distributed processing mode
 */
public final class DistributedStep4LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep4LocalInput(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cGetInput(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets an input key-value data collection object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step4LocalPartialModelsInputId id, KeyValueDataCollection val) {
        if (id != Step4LocalPartialModelsInputId.partialModels) {
            throw new IllegalArgumentException("Incorrect Step4LocalPartialModelsInputId");
        }
        cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input key-value data collection object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(Step4LocalPartialModelsInputId id) {
        if (id != Step4LocalPartialModelsInputId.partialModels) {
            throw new IllegalArgumentException("Incorrect Step4LocalPartialModelsInputId"); // error processing
        }
        return new KeyValueDataCollection(getContext(), cGetDataCollection(this.cObject, id.getValue()));
    }

    /**
     * Sets an input numeric table object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step4LocalNumericTableInputId id, NumericTable val) {
        if (id != Step4LocalNumericTableInputId.partialData &&
            id != Step4LocalNumericTableInputId.inputOfStep4FromStep2) {
            throw new IllegalArgumentException("Incorrect Step4LocalNumericTableInputId");
        }
        cSetNumericTable(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input numeric table object for the implicit ALS training algorithm
     * in the fourth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public NumericTable get(Step4LocalNumericTableInputId id) {
        if (id != Step4LocalNumericTableInputId.partialData &&
            id != Step4LocalNumericTableInputId.inputOfStep4FromStep2) {
            throw new IllegalArgumentException("Incorrect Step4LocalNumericTableInputId"); // error processing
        }
        if (id == Step4LocalNumericTableInputId.partialData) {
            return new CSRNumericTable(getContext(), cGetNumericTable(this.cObject, id.getValue()));
        } else /* if (id == Step4LocalNumericTableInputId.inputOfStep4FromStep2) */ {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(this.cObject, id.getValue()));
        }
    }

    private native long cGetInput(long cAlgorithm, int prec, int method);

    private native void cSetNumericTable(long cObject, int id, long numericTableAddr);
    private native long cGetNumericTable(long cObject, int id);

    private native void cSetDataCollection(long cObject, int id, long collectionAddr);
    private native long cGetDataCollection(long cObject, int id);
}
/** @} */
