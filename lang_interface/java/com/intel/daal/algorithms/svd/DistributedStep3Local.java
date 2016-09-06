/* file: DistributedStep3Local.java */
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

package com.intel.daal.algorithms.svd;

import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SVD__DISTRIBUTEDSTEP3LOCAL"></a>
 * @brief Runs the third step of the SVD algorithm in the distributed processing mode
 * \n<a href="DAAL-REF-SVD-ALGORITHM">SVD algorithm description and usage models</a>
 *
 * @par References
 *      - Method class.  SVD computation methods
 *      - DistributedStep3LocalInputId class. Identifiers of SVD input objects
 *      - ResultId class. Identifiers of SVD results
 *      - ResultFormat class. Options to return SVD output matrices
 *      - Parameter class
 *      - DistributedStep3LocalInput class
 *      - DistributedStep3LocalPartialResult class
 *      - Result class
 */
public class DistributedStep3Local extends AnalysisDistributed {
    public DistributedStep3LocalInput          input;        /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public Method                               method;  /*!< Computation method for the algorithm */
    private Result    result;      /*!< %Result of the algorithm */
    private DistributedStep3LocalPartialResult partialResult;   /*!< Partial result of the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the SVD algorithm by copying input objects and parameters
     * of another SVD algorithm
     * @param context   Context to manage created SVD algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep3Local(DaalContext context, DistributedStep3Local other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new DistributedStep3LocalInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the SVD algorithm
     * @param context   Context to manage created SVD algorithm
     * @param cls       Data type to use in intermediate computations for the SVD algorithm,
     *                  Double.class or Float.class
     * @param method    SVD computation method, @ref Method
     */
    public DistributedStep3Local(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;
        if (this.method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInitDistributed(prec.getValue(), method.getValue());
        input = new DistributedStep3LocalInput(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the SVD algorithm
     * @return  Partial results of the SVD algorithm obtained in the third step in the distributed processing mode
     */
    @Override
    public DistributedStep3LocalPartialResult compute() {
        super.compute();
        partialResult = new DistributedStep3LocalPartialResult(getContext(), cGetPartialResult(cObject, prec.getValue(), method.getValue()));
        result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return partialResult;
    }

    /**
     * Computes final results of the SVD algorithm
     * @return  Final results of the SVD algorithm
     */
    @Override
    public Result finalizeCompute() {
        return result;
    }

    /**
     * Returns the newly allocated SVD algorithm
     * with a copy of input objects and parameters of this SVD algorithm
     * @param context   Context to manage created SVD algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep3Local clone(DaalContext context) {
        return new DistributedStep3Local(context, this);
    }

    private native long cInitDistributed(int prec, int method);

    private native long cInitParameter(long addr, int prec, int method);

    private native long cGetInput(long addr, int prec, int method);

    private native long cGetResult(long addr, int prec, int method);

    private native long cGetPartialResult(long addr, int prec, int method);

    private native long cClone(long algAddr, int prec, int method);
}
