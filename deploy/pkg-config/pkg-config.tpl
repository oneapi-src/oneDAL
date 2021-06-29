#===============================================================================
# Copyright 2021 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

prefix=prefix=${{pcfiledir}}/../../
exec_prefix=${{prefix}}
libdir=${{exec_prefix}}/{libdir}
includedir=${{prefix}}/include

#info
Name: oneDAL
Description: Intel(R) oneAPI Data Analytics Library
Version: [dal.version]
URL: https://software.intel.com/en-us/oneapi/onedal
#Link line
Libs: {libs}
#Compiler line
Cflags: {opts}
