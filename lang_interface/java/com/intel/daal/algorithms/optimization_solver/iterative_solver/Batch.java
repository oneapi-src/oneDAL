/* file: Batch.java */
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

package com.intel.daal.algorithms.optimization_solver.iterative_solver;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__BATCH"></a>
 * @brief %Base interface for the iterative solver algorithm in the batch processing mode
 * \n<a href="DAAL-REF-ITERATIVE_SOLVER-ALGORITHM">iterative solver algorithm description and usage models</a>
 *
 * @par References
 *      - Parameter class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.InputId class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.ResultId class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.Input class
 *      - com.intel.daal.algorithms.optimization_solver.iterative_solver.Result class
 *
 */
public class Batch extends com.intel.daal.algorithms.optimization_solver.Batch {

    public Input input;     /*!< %Input data */
    public Parameter parameter; /*!< Parameters of the algorithm */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the iterative solver algorithm by copying input objects and parameters of another iterative solver algorithm
     * @param context    Context to manage the iterative solver algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);

        this.cObject = cClone(other.cObject);
        input = new Input(context, cGetInput(cObject));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__BATCH__BATCH"></a>
     * Constructs the iterative solver algorithm
     *
     * @param context      Context to manage the iterative solver algorithm
     */
    public Batch(DaalContext context) {
        super(context);
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHMS__OPTIMIZATION_SOLVER__ITERATIVE_SOLVER__BATCH__BATCH"></a>
     * Constructs the iterative solver algorithm
     *
     * @param context      Context to manage the iterative solver algorithm
     * @param cAlgorithm   Pointer to the C++ implememntation
     */
    public Batch(DaalContext context, long cAlgorithm) {
        super(context);

        this.cObject = cAlgorithm;
        input = new Input(context, cGetInput(cObject));
        parameter = new Parameter(getContext(), cGetParameter(this.cObject));
    }

    /**
     * Computes the iterative solver in the batch processing mode
     * @return  Results of the computation
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject));
        return result;
    }

    /**
     * Returns the newly allocated iterative solver algorithm
     * with a copy of input objects and parameters of this iterative solver algorithm
     * @param context    Context to manage the iterative solver algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit();
    private native long cClone(long algAddr);
    private native long cGetInput(long cAlgorithm);
    private native long cGetParameter(long cAlgorithm);
    protected native long cGetResult(long cAlgorithm);
}
