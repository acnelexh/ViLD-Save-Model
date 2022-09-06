import pandas as pd

def get_average(df, column = 1):
    sum = 0
    for i, column in enumerate(df[column]):
        if i % 4 == 0:
            sum += column

    return sum/11

objects = ['car', 'cyclist', 'pedestrian']

for object in objects:
    df = pd.read_csv(f'/data/datasets/kitti/evaluations/base/plot/{object}_detection.txt', delim_whitespace=True, header=None)

    columns = [1, 2, 3]
    ap = []
    for column in columns:
        ap.append(get_average(df, column))
    

    print(ap)
