/* file: TrainingInput.java */
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

package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.Model;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__TRAININGINPUT"></a>
 * @brief %Input objects for the implicit ALS training algorithm in the batch processing mode
 */
public class TrainingInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
        return new HomogenNumericTable(getContext(), cGetInput(this.cObject, id.getValue()));
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
