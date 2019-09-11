#! /bin/bash
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
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

# Script for building SWIG on Windows with MinGW #
##################################################

# You must have the following MinGW packages installed:
#	 mingw-developer-toolkit
#	 mingw32-base
#	 mingw32-gcc-g++
#	 msys-base

# Make sure you have the following env variable set:
#    FTP_PROXY=http://proxy-chain.intel.com:911

# Clone the daal repo and and make sure the daal/swig/swig submodule
# contains the swig files.  You may need to run:
#    `git submodule init`
#    `git submodule update`

# Make sure the swig submodule is on the correct branch

# Run this script in a MinGW shell (C:\MinGW\msys\1.0\msys.bat)

# If it complains about missing commands, add C:\MinGW\bin to your PATH.

cd swig

curl -o pcre-8.38.tar.gz ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/pcre-8.38.tar.gz

Tools/pcre-build.sh

./autogen.sh
./configure
make
