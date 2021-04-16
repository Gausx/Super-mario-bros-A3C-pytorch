import json


def rule():
    path = 'rule.txt'
    f = open(path)
    line = f.readline()

    a = []
    while line:
        a.extend(line.split('\n')[0].split(' '))
        line = f.readline()

    dict = {}
    for key in a:
        dict[key] = dict.get(key, 0) + 1

    print(dict)
    b = json.dumps(dict)
    f2 = open('new_json.json', 'w')
    f2.write(b)
    f2.close()


if __name__ == '__main__':
    rule()
