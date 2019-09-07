import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    class used to preprocess/clean data before model training
    reads data from raw directory , transform it and save it to processed directory
    """

    imageheight = 48
    imagewidth = 48
    imagechannels = 1
    processed_data_path = '../../data/processed/'

    def __init__(self):
        pass

    def load_data(self, path):
        """
        data loader function
        :param path: dataset path .csv file
        :return:pandas data frame object
        """
        chks = pd.read_csv(path, chunksize=1000)
        chlist = []
        for chk in chks:
            chlist.append(chk)
        df = pd.concat(chlist, axis=0)
        return df

    @staticmethod
    def str2img(row):
        """

        :param row: string of numbers
        :return: 3d array with shape width * height * channels (image)
        """
        return np.array([int(i) for i in row.split(' ')], dtype=np.uint8).reshape(DataPreprocessor.imagewidth,
                                                                                  DataPreprocessor.imageheight,
                                                                                  DataPreprocessor.imagechannels)

    def transform_raw_data(self, df):
        """
        transform every row of the FER2013 dataset into image and save images , emotions to processed data directory
        :param df:
        :return:None
        """
        pd.DataFrame(df['pixels'].apply(lambda x: DataPreprocessor.str2img(x))).to_pickle(
            DataPreprocessor.processed_data_path + 'images.pkl')
        pd.DataFrame(df['emotion']).to_pickle(DataPreprocessor.processed_data_path + 'emotions.pkl')
        print('Transformation done :)')
        return None


processor = DataPreprocessor()
df = processor.load_data('../../data/raw/fer2013.csv')
processor.transform_raw_data(df)
