#!/bin/bash
echo
echo "Testing started `date +'%d.%m.%y %H:%M:%S'`."
echo

file_algs=algorithms.txt

while read ALGORITHM_NAME
    do 
        echo Testing $ALGORITHM_NAME
        echo
        source ./tools/${ALGORITHM_NAME}/download_test.sh &>> _log_downloads
        python tools/${ALGORITHM_NAME}/patcher.py test_$ALGORITHM_NAME.py
        pytest -s --disable-warnings test_$ALGORITHM_NAME.py > _log.txt
        python tools/get_conformance_general.py

        if [ -f tools/${ALGORITHM_NAME}/get_conformance_local.py ]
        then
            python tools/${ALGORITHM_NAME}/get_conformance_local.py
            echo
        fi
        rm test_$ALGORITHM_NAME.py
        rm __n_calls.tmp
        rm _log*

done < $file_algs

echo
echo "Testing finished `date +'%d.%m.%y %H:%M:%S'`."