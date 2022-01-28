import sys
import pandas as pd
from sqlalchemy import create_engine
    
def load_data(messages_filepath, categories_filepath):
    """
    :param messages_filepath: the message filepath
    :param categories_filepath: the categories filepath
    :return: the merged dataframe

    This function reads the csv files and does some preliminary cleaning and merges the two
    dataframes into df
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.drop_duplicates(subset='id', inplace=True)
    categories = pd.read_csv(categories_filepath)
    categories.drop_duplicates(subset='id', inplace=True)
    # merge datasets
    df = messages.merge(categories, on='id', how='inner')
    df.fillna(value='no_original_message', inplace=True)
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:-2])
    print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # There are some values in "related" column equal to 2. Change them to 1.
    categories['related'] = categories['related'].astype(bool)  # change non-zeros to True
    categories['related'] = categories['related'].astype(int)  # change back to int format
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    return df


def clean_data(df):
    """
    :param df:
    :return: returns the cleaned dataframe

    This function removes duplicate lines from the dataframe
    """
    # drop duplicates
    print('length of df before duplicate=', len(df))
    df = df.drop_duplicates()
    # check number of duplicates
    print('length of df after duplicate=', len(df))
    print(df)
    return df

def save_data(df, database_filename):
    """

    :param df: the cleaned dataframe
    :param database_filename: filepath to the database
    :return:

    This function saves the cleaned dataframe df into a sqlite database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()