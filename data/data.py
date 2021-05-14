import pandas as pd


def load_wine_quality(type: str ='red') -> pd.DataFrame:
    """Loads the wine quality data from the UCI Machine Learnind Database.
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
        pd.DataFrame: Wine quality dataset
    """
    
    assert type in ['red', 'white', 'both']

    if type != 'white':
        try:
            df_red = pd.read_feather('wine_quality_red.feather')
        except FileNotFoundError:
            _download_wine_data()
            df_red = pd.read_feather('wine_quality_red.feather')
        if type == 'red':
            return df_red
    if type != 'red':
        try:
            df_white = pd.read_feather('wine_quality_white.feather')
        except FileNotFoundError:
            _download_wine_data()
            df_white = pd.read_feather('wine_quality_white.feather')
        if type == 'white':
            return df_white
    
    df_red['Type'] = 'Red'
    df_white['Type'] = 'White'

    return pd.concat([df_red, df_white])


def _download_wine_data():

    
    def download_df(type):
        assert type in ['red', 'white']
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-' + type + '.csv'
        df = pd.read_csv(url, sep=';')
        df.rename(str.title, axis='columns', inplace=True)
        df.rename(columns={'Ph': 'pH'}, inplace=True)
        return df

    df_red = download_df('red')
    df_red.to_feather('wine_quality_red.feather', compression='lz4', version=2)
    df_white = download_df('white')
    df_white.to_feather('wine_quality_white.feather', compression='lz4', version=2)