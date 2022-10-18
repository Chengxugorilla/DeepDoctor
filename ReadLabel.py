import pandas

test = pandas.read_excel('C:/Users/Amax/Downloads/001-870上颌窦侧壁骨厚度汇总.xlsx', usecols=['开窗骨壁宽度']).to_numpy()
i = 0
labels = []

for number in test:
        labels.extend(number)


