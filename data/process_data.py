import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    load data from csv files and merge into single dataframe
    
    Input:
    messages_filepath       filepath to messages csv file
    categories_filepath     filepath to categories csv file
    
    Returns:
    df                      dataframe merging categories and messages
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='left')
    return df


def clean_data(df):
    '''
    clean_data
    dataframe cleaned of stop words, duplicates, nulls and
    unnecessary data
    
    Input:
    df       merged dataframe
    
    Returns:
    df       cleaned dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # convert erroneous 2 values to 1
    categories['related'] = categories['related'].map(lambda x: 1 if x == 2 else x)
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # drop unnecessary column with nulls from dataset
    df = df.drop('original', axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    save_data
    saves dataframe `df` as `database_filename` 
    
    Input:
    df                    merged and cleaned dataframe
    database_filename     name to save database file
    
    Returns:
    
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterData', engine, index=False, if_exists='replace')


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