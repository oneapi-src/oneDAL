import re
import sys

fil_to_path = sys.argv[1]
dictionary = [('from sklearn.tree import DecisionTreeClassifier', 'from wrappers.wrapper_tree import DecisionTreeClassifier'),
              ('from sklearn.tree import DecisionTreeRegressor', 'from wrappers.wrapper_tree import DecisionTreeRegressor')]

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


