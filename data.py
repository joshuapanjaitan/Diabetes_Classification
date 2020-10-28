import pandas as pd


def to2D_List(dataframe, header):
    train = []
    lines = []
    id_vec = dataframe['age'].tolist()
    for i in range(len(id_vec)):
        for head in header:
            tmp = dataframe[head].tolist()
            lines.append(tmp[i])
        train.append(lines)
        lines = []
    return train


def writeRess(algo, pocket, id_vec):
    storage = [['ID', 'Class']]  # set the label
    for i in range(len(pocket)):  # compose
        tmp = [id_vec[i], pocket[i]]
        storage.append(tmp)
        tmp = []
    f = open(algo+'.csv', 'w')
    for item in storage:
        for i in range(len(item)):
            if i == 0:
                f.write(str(item[i]))
            else:
                f.write(',' + str(item[i]))
        f.write('\n')
    f.close()
