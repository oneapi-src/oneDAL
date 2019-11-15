/* file: SerializationTag.java */
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
 * @brief Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) package
 */
package com.intel.daal;

import com.intel.daal.utils.*;

/**
 * @ingroup serialization
 * @{
 */
/**
 * <a name="DAAL-CLASS-SERIALIZATIONTAG"></a>
 */
public final class SerializationTag {
    private int _value;

    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the serialization tag object using the provided value
     * @param value     Value corresponding to the serialization tag object
     */
    public SerializationTag(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the serialization tag object
     * @return Value corresponding to the serialization tag object
     */
    public int getValue() {
        return _value;
    }

    private static final int SERIALIZATION_HOMOGEN_FLOAT32_NT_ID_VALUE                             = 1000;
    private static final int SERIALIZATION_HOMOGEN_FLOAT64_NT_ID_VALUE                             = 1001;
    private static final int SERIALIZATION_HOMOGEN_INT32_S_NT_ID_VALUE                             = 1002;
    private static final int SERIALIZATION_HOMOGEN_INT32_U_NT_ID_VALUE                             = 1003;
    private static final int SERIALIZATION_HOMOGEN_INT64_S_NT_ID_VALUE                             = 1004;
    private static final int SERIALIZATION_HOMOGEN_INT64_U_NT_ID_VALUE                             = 1005;
    private static final int SERIALIZATION_AOS_NT_ID_VALUE                                         = 3000;
    private static final int SERIALIZATION_SOA_NT_ID_VALUE                                         = 3001;
    private static final int SERIALIZATION_DATACOLLECTION_ID_VALUE                                 = 4000;
    private static final int SERIALIZATION_KEYVALUEDATACOLLECTION_ID_VALUE                         = 4010;
    private static final int SERIALIZATION_MATRIX_NT_ID_VALUE                                      = 7000;
    private static final int SERIALIZATION_CSR_NT_ID_VALUE                                         = 8000;
    private static final int SERIALIZATION_JAVANIO_CSR_NT_ID_VALUE                                 = 9000;
    private static final int SERIALIZATION_JAVANIO_HOMOGEN_NT_ID_VALUE                             = 10010;
    private static final int SERIALIZATION_JAVANIO_AOS_NT_ID_VALUE                                 = 10020;
    private static final int SERIALIZATION_JAVANIO_SOA_NT_ID_VALUE                                 = 10030;
    private static final int SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID_VALUE                     = 10040;
    private static final int SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID_VALUE                    = 10050;
    private static final int SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID_VALUE                      = 101610;
    private static final int SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID_VALUE = 101620;
    private static final int SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID_VALUE         = 101630;
    private static final int SERIALIZATION_SVM_MODEL_ID_VALUE                                      = 100800;
    private static final int SERIALIZATION_UPPERPACKEDSYMMETRIC_FLOAT32_NT_ID_VALUE                = 11000;
    private static final int SERIALIZATION_UPPERPACKEDSYMMETRIC_FLOAT64_NT_ID_VALUE                = 11001;
    private static final int SERIALIZATION_UPPERPACKEDSYMMETRIC_INT32_S_NT_ID_VALUE                = 11002;
    private static final int SERIALIZATION_UPPERPACKEDSYMMETRIC_INT32_U_NT_ID_VALUE                = 11003;
    private static final int SERIALIZATION_UPPERPACKEDSYMMETRIC_INT64_S_NT_ID_VALUE                = 11004;
    private static final int SERIALIZATION_UPPERPACKEDSYMMETRIC_INT64_U_NT_ID_VALUE                = 11005;
    private static final int SERIALIZATION_LOWERPACKEDSYMMETRIC_FLOAT32_NT_ID_VALUE                = 11020;
    private static final int SERIALIZATION_LOWERPACKEDSYMMETRIC_FLOAT64_NT_ID_VALUE                = 11021;
    private static final int SERIALIZATION_LOWERPACKEDSYMMETRIC_INT32_S_NT_ID_VALUE                = 11022;
    private static final int SERIALIZATION_LOWERPACKEDSYMMETRIC_INT32_U_NT_ID_VALUE                = 11023;
    private static final int SERIALIZATION_LOWERPACKEDSYMMETRIC_INT64_S_NT_ID_VALUE                = 11024;
    private static final int SERIALIZATION_LOWERPACKEDSYMMETRIC_INT64_U_NT_ID_VALUE                = 11025;
    private static final int SERIALIZATION_UPPERPACKEDTRIANGULAR_FLOAT32_NT_ID_VALUE               = 12000;
    private static final int SERIALIZATION_UPPERPACKEDTRIANGULAR_FLOAT64_NT_ID_VALUE               = 12001;
    private static final int SERIALIZATION_UPPERPACKEDTRIANGULAR_INT32_S_NT_ID_VALUE               = 12002;
    private static final int SERIALIZATION_UPPERPACKEDTRIANGULAR_INT32_U_NT_ID_VALUE               = 12003;
    private static final int SERIALIZATION_UPPERPACKEDTRIANGULAR_INT64_S_NT_ID_VALUE               = 12004;
    private static final int SERIALIZATION_UPPERPACKEDTRIANGULAR_INT64_U_NT_ID_VALUE               = 12005;
    private static final int SERIALIZATION_LOWERPACKEDTRIANGULAR_FLOAT32_NT_ID_VALUE               = 12020;
    private static final int SERIALIZATION_LOWERPACKEDTRIANGULAR_FLOAT64_NT_ID_VALUE               = 12021;
    private static final int SERIALIZATION_LOWERPACKEDTRIANGULAR_INT32_S_NT_ID_VALUE               = 12022;
    private static final int SERIALIZATION_LOWERPACKEDTRIANGULAR_INT32_U_NT_ID_VALUE               = 12023;
    private static final int SERIALIZATION_LOWERPACKEDTRIANGULAR_INT64_S_NT_ID_VALUE               = 12024;
    private static final int SERIALIZATION_LOWERPACKEDTRIANGULAR_INT64_U_NT_ID_VALUE               = 12025;
    private static final int SERIALIZATION_MERGE_NT_ID_VALUE                                       = 13000;
    private static final int SERIALIZATION_ROWMERGE_NT_ID_VALUE                                    = 14000;

    public static final SerializationTag SERIALIZATION_HOMOGEN_FLOAT32_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_FLOAT32_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_FLOAT64_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_FLOAT64_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT32_S_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_INT32_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT32_U_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_INT32_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT64_S_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_INT64_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_HOMOGEN_INT64_U_NT_ID     = new SerializationTag(SERIALIZATION_HOMOGEN_INT64_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_AOS_NT_ID                 = new SerializationTag(SERIALIZATION_AOS_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_SOA_NT_ID                 = new SerializationTag(SERIALIZATION_SOA_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_DATACOLLECTION_ID         = new SerializationTag(SERIALIZATION_DATACOLLECTION_ID_VALUE);
    public static final SerializationTag SERIALIZATION_KEYVALUEDATACOLLECTION_ID = new SerializationTag(SERIALIZATION_KEYVALUEDATACOLLECTION_ID_VALUE);
    public static final SerializationTag SERIALIZATION_MATRIX_NT_ID              = new SerializationTag(SERIALIZATION_MATRIX_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_CSR_NT_ID                 = new SerializationTag(SERIALIZATION_CSR_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_JAVANIO_CSR_NT_ID         = new SerializationTag(SERIALIZATION_JAVANIO_CSR_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_JAVANIO_HOMOGEN_NT_ID     = new SerializationTag(SERIALIZATION_JAVANIO_HOMOGEN_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_JAVANIO_AOS_NT_ID         = new SerializationTag(SERIALIZATION_JAVANIO_AOS_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_JAVANIO_SOA_NT_ID         = new SerializationTag(SERIALIZATION_JAVANIO_SOA_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID  = new SerializationTag(SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID = new SerializationTag(SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_SVM_MODEL_ID              = new SerializationTag(SERIALIZATION_SVM_MODEL_ID_VALUE);
    public static final SerializationTag SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID                      = new SerializationTag(SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID_VALUE);
    public static final SerializationTag SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID         = new SerializationTag(SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID = new SerializationTag(SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDSYMMETRIC_FLOAT32_NT_ID  = new SerializationTag(SERIALIZATION_UPPERPACKEDSYMMETRIC_FLOAT32_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDSYMMETRIC_FLOAT64_NT_ID  = new SerializationTag(SERIALIZATION_UPPERPACKEDSYMMETRIC_FLOAT64_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDSYMMETRIC_INT32_S_NT_ID  = new SerializationTag(SERIALIZATION_UPPERPACKEDSYMMETRIC_INT32_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDSYMMETRIC_INT32_U_NT_ID  = new SerializationTag(SERIALIZATION_UPPERPACKEDSYMMETRIC_INT32_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDSYMMETRIC_INT64_S_NT_ID  = new SerializationTag(SERIALIZATION_UPPERPACKEDSYMMETRIC_INT64_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDSYMMETRIC_INT64_U_NT_ID  = new SerializationTag(SERIALIZATION_UPPERPACKEDSYMMETRIC_INT64_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDSYMMETRIC_FLOAT32_NT_ID  = new SerializationTag(SERIALIZATION_LOWERPACKEDSYMMETRIC_FLOAT32_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDSYMMETRIC_FLOAT64_NT_ID  = new SerializationTag(SERIALIZATION_LOWERPACKEDSYMMETRIC_FLOAT64_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDSYMMETRIC_INT32_S_NT_ID  = new SerializationTag(SERIALIZATION_LOWERPACKEDSYMMETRIC_INT32_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDSYMMETRIC_INT32_U_NT_ID  = new SerializationTag(SERIALIZATION_LOWERPACKEDSYMMETRIC_INT32_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDSYMMETRIC_INT64_S_NT_ID  = new SerializationTag(SERIALIZATION_LOWERPACKEDSYMMETRIC_INT64_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDSYMMETRIC_INT64_U_NT_ID  = new SerializationTag(SERIALIZATION_LOWERPACKEDSYMMETRIC_INT64_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDTRIANGULAR_FLOAT32_NT_ID = new SerializationTag(SERIALIZATION_UPPERPACKEDTRIANGULAR_FLOAT32_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDTRIANGULAR_FLOAT64_NT_ID = new SerializationTag(SERIALIZATION_UPPERPACKEDTRIANGULAR_FLOAT64_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDTRIANGULAR_INT32_S_NT_ID = new SerializationTag(SERIALIZATION_UPPERPACKEDTRIANGULAR_INT32_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDTRIANGULAR_INT32_U_NT_ID = new SerializationTag(SERIALIZATION_UPPERPACKEDTRIANGULAR_INT32_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDTRIANGULAR_INT64_S_NT_ID = new SerializationTag(SERIALIZATION_UPPERPACKEDTRIANGULAR_INT64_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_UPPERPACKEDTRIANGULAR_INT64_U_NT_ID = new SerializationTag(SERIALIZATION_UPPERPACKEDTRIANGULAR_INT64_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDTRIANGULAR_FLOAT32_NT_ID = new SerializationTag(SERIALIZATION_LOWERPACKEDTRIANGULAR_FLOAT32_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDTRIANGULAR_FLOAT64_NT_ID = new SerializationTag(SERIALIZATION_LOWERPACKEDTRIANGULAR_FLOAT64_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDTRIANGULAR_INT32_S_NT_ID = new SerializationTag(SERIALIZATION_LOWERPACKEDTRIANGULAR_INT32_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDTRIANGULAR_INT32_U_NT_ID = new SerializationTag(SERIALIZATION_LOWERPACKEDTRIANGULAR_INT32_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDTRIANGULAR_INT64_S_NT_ID = new SerializationTag(SERIALIZATION_LOWERPACKEDTRIANGULAR_INT64_S_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_LOWERPACKEDTRIANGULAR_INT64_U_NT_ID = new SerializationTag(SERIALIZATION_LOWERPACKEDTRIANGULAR_INT64_U_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_MERGE_NT_ID                         = new SerializationTag(SERIALIZATION_MERGE_NT_ID_VALUE);
    public static final SerializationTag SERIALIZATION_ROWMERGE_NT_ID                      = new SerializationTag(SERIALIZATION_ROWMERGE_NT_ID_VALUE);
}
/** @} */
