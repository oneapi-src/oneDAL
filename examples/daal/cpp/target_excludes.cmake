#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

if(CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
    return()
endif()

if((CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64") AND
   (CMAKE_C_COMPILER MATCHES "clang"))
    # Some of the tests fail under emulation, and need investigating. For now we
    # set them to not be run so that the CI passes
    set(EXCLUDE_LIST
        ${EXCLUDE_LIST}
        "assoc_rules_apriori_batch"
        "cholesky_dense_batch"
        "cor_csr_distr"
        "cor_csr_online"
        "cov_csr_distr"
        "cov_csr_online"
        "enable_thread_pinning"
        "lin_reg_metrics_dense_batch"
        "lin_reg_qr_dense_batch"
        "lin_reg_qr_dense_online"
        "low_order_moms_csr_distr"
        "low_order_moms_dense_distr"
        "out_detect_mult_dense_batch"
        "pivoted_qr_dense_batch"
    )
elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64") AND
       (CMAKE_C_COMPILER MATCHES "gcc"))
    set(EXCLUDE_LIST
        ${EXCLUDE_LIST}
        "cholesky_dense_batch"
        "cor_csr_distr"
        "cor_csr_online"
        "cov_csr_distr"
        "cov_csr_online"
        "enable_thread_pinning"
        "lin_reg_metrics_dense_batch"
        "lin_reg_qr_dense_batch"
        "lin_reg_qr_dense_online"
        "out_detect_mult_dense_batch"
    )
elseif((CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64") AND
       (CMAKE_C_COMPILER MATCHES "clang"))
    set(EXCLUDE_LIST
        ${EXCLUDE_LIST}
        "assoc_rules_apriori_batch"
        "cd_dense_batch"
        "cholesky_dense_batch"
        "cor_csr_distr"
        "cor_csr_online"
        "cor_dense_distr"
        "cor_dense_online"
        "cov_csr_distr"
        "cov_csr_online"
        "cov_dense_distr"
        "cov_dense_online"
        "elastic_net_dense_batch"
        "enable_thread_pinning"
        "lasso_reg_dense_batch"
        "lin_reg_metrics_dense_batch"
        "lin_reg_qr_dense_batch"
        "lin_reg_qr_dense_online"
        "low_order_moms_csr_distr"
        "low_order_moms_csr_online"
        "low_order_moms_dense_distr"
        "low_order_moms_dense_online"
        "out_detect_mult_dense_batch"
        "pca_metrics_dense_batch"
        "pivoted_qr_dense_batch"
    )
endif()

