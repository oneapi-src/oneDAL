import re
import sys, os
from datetime import datetime
from subprocess import Popen, PIPE
from utils import make_report

algs_filename = "algorithms.txt"
report_filename = "report.html"

if __name__ == "__main__":
    with open(algs_filename, "r") as file_algs:
        algs = file_algs.read().split("\n")
    algs.remove("")

    print("Confromance testing start")
    for alg_name in algs:
        os.system("bash ./download_tests.sh --alg-name %s --scikit-version 0.23.1 &>> _log_downloads" % alg_name)
        print(alg_name)
        os.system("python ./patcher_of_tests.py %s" % alg_name)

        os.system("IDP_SKLEARN_VERBOSE=INFO python -m daal4py -m pytest -s --disable-warnings test_%s.py > _log_%s.txt" % (alg_name,alg_name))

    make_report(algs_filename=algs_filename,
                report_filename = report_filename)
