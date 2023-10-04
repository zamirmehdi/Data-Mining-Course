import how as how
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

dataset_path = '/Users/apple/Documents/iris.data'
df = pd.read_csv(dataset_path, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])


def print_nans():
    print('-Part1:\n')
    # Find and show NaNs
    print('Number of sepal_length NaNs:', len(df[df['sepal_length'].isna()]))
    print(df[df['sepal_length'].isna()])
    # print(len(df['sepal_length'].dropna()))

    # print((df['sepal_length'].isna()).where(df['sepal_length'] == True))
    # print_hi('PyCharm')

    print('\n\nNumber of sepal_width NaNs:', len(df[df['sepal_width'].isna()]))
    print(df[df['sepal_width'].isna()])

    print('\n\nNumber of petal_length NaNs:', len(df[df['petal_length'].isna()]))
    print(df[df['petal_length'].isna()])

    print('\n\nNumber of petal_width NaNs:', len(df[df['petal_width'].isna()]))
    print(df[df['petal_width'].isna()])

    print('\n\nNumber of target NaNs:', len(df[df['target'].isna()]))
    print(df[df['target'].isna()])


def remove_nans():
    # print(df[df.isna()])
    global df
    print('\n\nNumber of rows = ', len(df))
    df = df.dropna()
    df.reset_index(drop=True)
    print('NaNs were removed. Number of rows = ', len(df))


def label_encoder():
    print('\n\n-Part2:\n')
    global df
    print('Before encode:')
    print(df)
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])
    print('\nAfter encode:')
    print(df)


def calculate_mean_and_variance(dataframe):
    print('sepal_length: \n' +
          f'    mean= {dataframe["sepal_length"].mean()}' + ', ' + f'variance= {dataframe["sepal_length"].var()}')
    print('sepal_width: \n' +
          f'    mean= {dataframe["sepal_width"].mean()}' + ', ' + f'variance= {dataframe["sepal_width"].var()}')
    print('petal_length: \n' +
          f'    mean= {dataframe["petal_length"].mean()}' + ', ' + f'variance= {dataframe["petal_length"].var()}')
    print('petal_width: \n' +
          f'    mean= {dataframe["petal_width"].mean()}' + ', ' + f'variance= {dataframe["petal_width"].var()}')


def normalization():
    print('\n\n-Part3:\n')
    global df
    print('Before normalization:')
    calculate_mean_and_variance(df)

    # box plot before normalization:
    new_data_set = pd.DataFrame(df, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
    boxplot = new_data_set.boxplot(column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    plt.show()

    standard_sc = StandardScaler()
    new_df = standard_sc.fit_transform(df.iloc[:, 0:-1])

    df_numpy = np.append(new_df, df[['target']].to_numpy(), axis=1)
    df = pd.DataFrame(df_numpy, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])

    # box plot after normalization:
    new_data_set = pd.DataFrame(df, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
    boxplot = new_data_set.boxplot(column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    plt.show()

    print('\nAfter normalization:')
    calculate_mean_and_variance(df)


def do_pca():
    global df
    print('\n\n-Part4:\n')
    print('After PCA:')
    df_target_col = df[['target']]
    data_set = df.iloc[:, 0:-1]
    pca = PCA(n_components=2)
    df = pca.fit_transform(data_set)
    new_df = pd.DataFrame(data=df, columns=['col 1', 'col 2'])
    new_df = pd.concat([new_df, df_target_col], axis=1)
    new_df = new_df.dropna()
    df = new_df
    print(new_df)
    visualize(df)


def visualize(data_set):
    # total plot:
    colors = {'0': 'red', '1': 'grey', '2': 'yellow'}
    fig, ax = plt.subplots()

    for i in range(len(data_set['col 1'])):
        if i in list(data_set['target'].index):
            ax.scatter(data_set['col 1'][i], data_set['col 2'][i],
                       color=colors[str(int(data_set['target'][i]))])
    ax.set_xlabel('sepal_length')
    ax.set_ylabel('sepal_width')

    ax.legend()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_nans()
    remove_nans()
    label_encoder()
    normalization()
    do_pca()
