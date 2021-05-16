import os
import pandas as pd

DATA_DIR = os.path.dirname(os.path.realpath(__file__))

def load_wine_quality(type: str ='red') -> pd.DataFrame:
    """Loads the wine quality data from the UCI Machine Learning Database.
    The data is described [here](https://archive.ics.uci.edu/ml/datasets/wine+quality)
    There are eleven features:
    1 - Fixed Acidity
    2 - Volatile Acidity
    3 - Citric Acid
    4 - Residual Sugar
    5 - Chlorides
    6 - Free Sulfur Dioxide
    7 - Total Sulfur Dioxide
    8 - Density
    9 - pH
    10 - Sulphates
    11 - Alcohol

    and the target variable
    12 - Quality (score between 0 and 10)

    If type='both', an additional column 'Type' is added to distinguish between red and white wine.

    The red wine data has 1599 observations and the white wine data has 4898 data for a total of 6497 observations.
    There are no missing values in the data.

    Args:
        type (str, optional): Which data to return, must be 'red', 'white', or 'both'. Defaults to 'red'.

    Returns:
        pd.DataFrame: If `type` is `'red'` or `'white'` a single DataFrame is returned. For `type='both'`, a tuple of DataFrames `(df_red, df_white)` is returned
    """
    
    assert type in ['red', 'white', 'both'], "type has to be either 'red', 'white', or 'both'"

    if type != 'white':
        try:
            df_red = pd.read_feather(DATA_DIR + '/wine_quality_red.feather')
        except FileNotFoundError:
            _download_wine_data()
            df_red = pd.read_feather(DATA_DIR + '/wine_quality_red.feather')
        if type == 'red':
            return df_red
    if type != 'red':
        try:
            df_white = pd.read_feather(DATA_DIR + '/wine_quality_white.feather')
        except FileNotFoundError:
            _download_wine_data()
            df_white = pd.read_feather(DATA_DIR + '/wine_quality_white.feather')
        if type == 'white':
            return df_white

    return df_red, df_white


def _download_wine_data():

    def download_df(type):
        assert type in ['red', 'white']
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-' + type + '.csv'
        df = pd.read_csv(url, sep=';')
        df.rename(str.title, axis='columns', inplace=True)
        df.rename(columns={'Ph': 'pH'}, inplace=True)
        return df

    df_red = download_df('red')
    df_red.to_feather(DATA_DIR + '/wine_quality_red.feather', compression='lz4', version=2)
    df_white = download_df('white')
    df_white.to_feather(DATA_DIR + '/wine_quality_white.feather', compression='lz4', version=2)


def _download_adult_data():
    def download_df(type):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.' + type
        names = ['Age', 'Workclass', 'Final Weight', 'Education', 'Years of Education', 'Marital Status', 
        'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per Week', 
        'Native Country', 'Income']
        df = pd.read_csv(url, header=0, names=names)
        return df
    
    df_train = download_df('data')
    df_train.to_feather(DATA_DIR + '/adult_train.feather', compression='lz4', version=2)
    df_test = download_df('test')
    df_test.to_feather(DATA_DIR + '/adult_test.feather', compression='lz4', version=2)


def load_adult_data(type: str='both') -> pd.DataFrame:
    """Load adult data from the UCI Machine Learning Database.
    The data is described [here](https://archive.ics.uci.edu/ml/datasets/adult)

    There are 14 features

    Age: continuous.
    Workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    Final Weight: continuous.
    Education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    Years of Education: continuous.
    Marital Status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    Occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    Relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    Race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    Sex: Female, Male.
    Capital Gain: continuous.
    Capital Loss: continuous.
    Hours per Week: continuous.
    Native Country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.ge: continuous.

    and the target variable:
    Income: >= 50k, <50k

    Args:
        type (str, optional): Which data to load. Must be 'train', 'test', or 'both'. Defaults to 'both'.

    Returns:
        pd.DataFrame: If `type` is `'train'` or `'test'` a single DataFrame is returned, For `type='both'`, a tuple of DataFrames `(df_train, df_test)` is returned
    """
    
    assert type in ['train', 'test', 'both'], "type has to be either 'train', 'test', or 'both'"
    
    if type != 'test':
        try:
            df_train = pd.read_feather(DATA_DIR + '/adult_train.feather')
        except FileNotFoundError:
            _download_adult_data()
            df_train = pd.read_feather(DATA_DIR + '/adult_train.feather')
        if type=='train':
            return df_train
    
    if type != 'train':
        try:
            df_test = pd.read_feather(DATA_DIR + '/adult_test.feather')
        except FileNotFoundError:
            _download_adult_data()
            df_test = pd.read_feather(DATA_DIR + '/adult_test.feather')
        if type=='test':
            return df_test
    
    return df_train, df_test
