/* file: TrainingInput.java */
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
 * @ingroup implicit_als_training_batch
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.Model;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__TRAININGINPUT"></a>
 * @brief %Input objects for the implicit ALS training algorithm in the batch processing mode
 */
public class TrainingInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TrainingInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    public TrainingInput(DaalContext context, long cAlgorithm, Precision prec, TrainingMethod method) {
        super(context);
        this.cObject = cInit(cAlgorithm, prec.getValue(), method.getValue());
    }

    /**
     * Sets an input numeric table object for the implicit ALS training algorithm in the batch processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(NumericTableInputId id, NumericTable val) {
        if (id != NumericTableInputId.data) {
            throw new IllegalArgumentException("Incorrect NumericTableInputId");
        }
        cSetInput(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Sets an input model object for the implicit ALS training algorithm in the batch processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(ModelInputId id, Model val) {
        if (id != ModelInputId.inputModel) {
            throw new IllegalArgumentException("Incorrect ModelInputId");
        }
        cSetInput(this.cObject, id.getValue(), val.getCObject());
    }

    /**
     * Returns an input numeric table object for the implicit ALS training algorithm in the batch processing mode
     * @param id      Identifier of the input object
     * @return        Input object that corresponds to the given identifier
     */
    public NumericTable get(NumericTableInputId id) {
        if (id != NumericTableInputId.data) {
            throw new IllegalArgumentException("Incorrect NumericTableInputId"); // error processing
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetInput(this.cObject, id.getValue()));
    }

    /**
     * Returns an input model object for the implicit ALS training algorithm in the batch processing mode
     * @param id      Identifier of the input object
     * @return        Input object that corresponds to the given identifier
     */
    public Model get(ModelInputId id) {
        if (id != ModelInputId.inputModel) {
            throw new IllegalArgumentException("Incorrect ModelInputId"); // error processing
        }
        return new Model(getContext(), cGetInput(this.cObject, id.getValue()));
    }

    private native long cInit(long algAddr, int prec, int method);

    private native void cSetInput(long cObject, int id, long resAddr);

    private native long cGetInput(long cObject, int id);
}
/** @} */
