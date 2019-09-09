/* file: Parameter.java */
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
 * @ingroup coordinate_descent
 * @{
 */
/**
 * @brief Contains classes for computing Coordinate Descent algorithm
 */
package com.intel.daal.algorithms.optimization_solver.coordinate_descent;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.algorithms.optimization_solver.sum_of_functions.Batch;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__COORDINATE_DESCENT__PARAMETER"></a>
 * @brief Parameter of the Coordinate Descent algorithm
 */
public class Parameter extends com.intel.daal.algorithms.optimization_solver.iterative_solver.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the parameter for the Coordinate Descent algorithm
     * @param context       Context to manage the parameter for the Coordinate Descent algorithm
     */
    public Parameter(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for the Coordinate Descent algorithm
     * @param context    Context to manage the Coordinate Descent algorithm
     * @param cObject    Pointer to C++ implementation of the parameter
     */
    public Parameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
    * @DAAL_DEPRECATED
    * Sets the seed for random generation of 32 bit integer indices of terms in the objective function.
    * @param seed The seed for random generation of 32 bit integer indices of terms in the objective function.
    */
    public void setSeed(int seed) {
        cSetSeed(this.cObject, seed);
    }

    /**
     * @DAAL_DEPRECATED
     * Gets the seed for random generation of 32 bit integer indices of terms in the objective function.
     * @return The seed for random generation of 32 bit integer indices of terms in the objective function.
     */
    public int getSeed() {
        return cGetSeed(this.cObject);
    }

    /**
    * @DAAL_DEPRECATED
    * Sets the selection strategy for algorithm
    * @param selection The selection stategy can be cyclic=0 and random=1
    */
    public void setSelection(int selection) {
        cSetSelection(this.cObject, selection);
    }

    /**
     * @DAAL_DEPRECATED
     * Gets the selection strategy for algorithm
     * @return The selection stategy can be cyclic=0 and random=1
     */
    public int getSelection() {
        return cGetSelection(this.cObject);
    }

    /**
    * @DAAL_DEPRECATED
    * Sets the positive flag parameter of algorithm
    * @param flag The flag for positive parameter of algorithm
    */
    public void setPositiveFlag(boolean flag) {
        cSetPositiveFlag(this.cObject, flag);
    }

    /**
     * @DAAL_DEPRECATED
     * Gets the positive flag parameter of algorithm
     * @return The flag for positive parameter of algorithm
     */
    public boolean getPositiveFlag() {
        return cGetPositiveFlag(this.cObject);
    }

    /**
    * @DAAL_DEPRECATED
    * Sets the flag to skip the first component computation
    * @param flag The flag to skip computation of first components
    */
    public void setSkipTheFirstComponentsFlag(boolean flag) {
        cSetSkipTheFirstComponentsFlag(this.cObject, flag);
    }

    /**
     * @DAAL_DEPRECATED
     * Gets the flag to skip the first component computation
     * @return The flag to skip computation of first components
     */
    public boolean getSkipTheFirstComponentsFlag() {
        return cGetSkipTheFirstComponentsFlag(this.cObject);
    }


    /**
     * Sets the engine to be used by the algorithm
     * @param engine to be used by the algorithm
     */
    public void setEngine(com.intel.daal.algorithms.engines.BatchBase engine) {
        cSetEngine(cObject, engine.cObject);
    }

    private native void cSetSeed(long parAddr, int seed);
    private native int cGetSeed(long parAddr);
    private native void cSetEngine(long cObject, long cEngineObject);
    private native void cSetSelection(long parAddr, int selection);
    private native int cGetSelection(long parAddr);
    private native void cSetPositiveFlag(long parAddr, boolean flag);
    private native boolean cGetPositiveFlag(long parAddr);
    private native void cSetSkipTheFirstComponentsFlag(long parAddr, boolean flag);
    private native boolean cGetSkipTheFirstComponentsFlag(long parAddr);
}
/** @} */
