import re
import sys

fil_to_path = sys.argv[1]
dictionary = [('from sklearn.ensemble import AdaBoostClassifier', 'from wrappers.wrapper_adaboost import AdaBoostClassifier')]

def path_file(file_name, dictionary):
    result_str = ''

    f = open(file_name, "r")
    lines = f.readlines()
    f.close()

    for to_path_line, path_line in dictionary:
        for i in range(len(lines)):
            if to_path_line in lines[i]:
                lines[i] = path_line + '\n'
                break

    f = open(file_name, "w")
    f.writelines(lines)
    f.close()


if __name__ == "__main__":
    path_file(fil_to_path, dictionary)


