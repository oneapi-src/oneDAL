import re
import sys, os
from datetime import datetime
from subprocess import Popen, PIPE

if __name__ == "__main__":
    with open("algorithms.txt", "r") as file_algs:
        algs = file_algs.read().split("\n")
    algs.remove("")

    report_filename = "report.html"
    report_file = open(report_filename,'wt')

    textHTML = """<html>
        <head>
        <title>
            Report of conformance testing
        </title>
        </head>
        <body>
            <p>
            Start of testing in """ + str(datetime.now())+ "<br>"
    report_file.write(textHTML)

    #os.system("bash ./download_tests2.sh")
    #os.system("export IDP_SKLEARN_VERBOSE=INFO")

    for alg_name in algs:
        report_file.write("<br><h2>Testing %s</h2>" % alg_name)
        #print("alg_name: ", alg_name)
        #os.system("source ./tools/%s/download_test.sh &>> _log_downloads" % alg_name)
        os.system("bash ./download_tests.sh --alg-name %s --scikit-version 0.21.3" % alg_name)
        print(alg_name)

        os.system("python ./patcher_of_tests.py %s" % alg_name)
        #out, err = Popen("python tools/%(alg)s/patcher.py test_%(alg)s.py" % {"alg": alg_name}, shell=True, stdout=PIPE).communicate()
        #out, err = Popen("python ./patcher.py %(alg)s" % {"alg": alg_name}, shell=True, stdout=PIPE).communicate()
        #report_file.write((str(out, 'utf-8')).replace("\n", "<br>"))
        os.system("IDP_SKLEARN_VERBOSE=INFO python -m daal4py -m pytest -s --disable-warnings test_%s.py > _log_%s.txt" % (alg_name,alg_name))

        out, err = Popen("python ./log_parser.py %(alg)s" % {"alg": alg_name}, shell=True, stdout=PIPE).communicate()
        report_file.write((str(out, 'utf-8')).replace("\n", "<br>"))

        

        #out, err = Popen("python tools/get_conformance_general.py", shell=True, stdout=PIPE).communicate()
        #report_file.write((str(out, 'utf-8')).replace("\n", "<br>"))
        #if os.path.exists("tools/%s/get_conformance_local.py" % alg_name):
        #    out, err = Popen("python tools/%s/get_conformance_local.py" % alg_name, shell=True, stdout=PIPE).communicate()
        #    report_file.write((str(out, 'utf-8')).replace("\n", "<br>"))

        #os.system("rm test_%s.py" % alg_name)
        #os.system("rm __n_calls.tmp")
        #os.system("rm _log*")
    
    textHTML = """<br>
    Finishing testing in """+str(datetime.now())+"""<br></p>
        </body>
    </html>"""

    report_file.write(textHTML)
    report_file.close()
