# import libraries
import argparse
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

parser = argparse.ArgumentParser()
parser.add_argument('msg_path', help='Path to the disaster_messages.csv file')
parser.add_argument('cat_path', help='Path to the disaster_categories.csv file')
parser.add_argument('db_path', help='Path or name of the database where to store the data')
args = parser.parse_args()


def load_file(file_path):
    """
        Load data function.
        This function considers the extension of the target file to load.
        It supports:
        - .csv
        - .json
        - .xml
        - .db

        Inputs:
            file_path - relative file path including the file name and its extension
        Outputs:
            file_content - pandas DataFrame containing the data of the input file
    """

    # grabs the extension of the file provided as input
    input_suffix = Path(file_path).suffix

    # load data if file is a csv
    if input_suffix == '.csv':
        file_content = pd.read_csv(file_path)

    # load data if file is a json
    elif input_suffix == '.json':
        file_content = pd.read_json(file_path, orient='records')

    # load data if file is a xml
    elif input_suffix == '.xml':
        from bs4 import BeautifulSoup
        with open(file_path) as fp:
            soup = BeautifulSoup(fp, "lxml")  # lxml is the Parser type
            data_dict = {}
            # use the find_all method to get all record tags in the document
            for record in soup.find_all('record'):
                # use the find_all method to get all fields in each record
                for record in record.find_all('field'):
                    if record['name'] not in data_dict:
                        data_dict[record['name']] = []
                    data_dict[record['name']].append(record.text)

            file_content = pd.DataFrame.from_dict(data_dict)

    # load data if from a db
    elif input_suffix == '.db':
        input_engine = create_engine('sqlite:///{}'.format(file_path))
        table_name = input("Please provide the name of the table to use as a date source: ")
        if table_name:
            try:
                file_content = pd.read_sql('SELECT * FROM {}'.format(table_name), input_engine)
            except:
                raise
        else:
            table_name = input("Please provide the name of the table to use as a date source: ")

    return file_content


def create_feat_from_col(df, col, val_sep):
    """
        Creates new columns from a column containing multiple values.

        Inputs:
            df - the dataframe to look into
            col - the specific column containing multiple values to split
            val_sep - the separator to use to split the values

        Outputs:
            new_features -> a dataframe containing one column per value, the
            header being the values of the first row
    """

    # create a dataframe from the values in the col
    new_features = df[col].str.split(val_sep, expand=True)

    # use the first row of the dataframe to get the new columns names
    first_row = new_features.loc[1]
    features_name = [x[:-2] for x in first_row]

    # rename the columns of `categories`
    new_features.columns = features_name

    return new_features


def extract_value(df, split_char, char_to_get, to_num=True, cols=None):
    """
        Extracts specific characters from a string and uses this extracted content
        as the values of a column.

        Load data function.
        This function considers the extension of the target file to load.
        It supports:
        - .csv
        - .json
        - .xml
        - .db

        Inputs:
            df - the dataframe to look into
            split_char - the separator to use to split the values
            char_to_get - which value to keep from the splitting
            to_num - set to True by default, whether we want to convert
                to numeric the extracted value or not
            cols - set to None by default, which means all columns of the df
                should be looked at

        Outputs:
            new_df -> a dataframe containing the values extracted
    """

    new_df = df.copy()
    if cols is None:
        columns = df.columns

    else:
        columns = cols

    for column in columns:
        new_df[column] = new_df[column].str.split(split_char).str.get(char_to_get).astype(str)
        if to_num is True:
            new_df[column] = pd.to_numeric(new_df[column])

    return new_df


def store_to_db(database, df, table_name):
    """
        Stores the content of a dataframe to a target table in a database using
        sqlachemy.
        WARNING: If the table alread exists it is replaced by the new data.

        Inputs:
            database - the database to connect to
            df - the dataframe to load to the database
            table_name - the name of the table to create or update
        Outputs:
            engine -> the engine created to establish a connection to the database
            (returned for future use)
    """

    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    return engine


def main():
    if args.msg_path:
        if args.cat_path:
            if args.db_path:
                # load messages dataset
                messages = load_file('messages.csv')
                # load categories dataset
                categories = load_file('categories.csv')

                # create a feature per value of categories
                cat_df = create_feat_from_col(categories, 'categories', ';')
                cat_df = extract_value(cat_df, '-', 1)

                # merge the new category dataframe to the messages dataframe
                df = pd.concat([messages, cat_df], axis=1)
                print(df)

                # removes duplicates
                if df[df.duplicated()].shape[0] != 0:
                    df.drop_duplicates(inplace=True)

                # creates a table in the database with the dataframe
                # we get the engine to be able to use it afterwards
                engine = store_to_db(args.db_path, df, 'MessagesWithCategory')

            else:
                print('Please specify the path or name of the target database where to load the data')
        else:
            print('Please specify the path to the disaster categories file')
    else:
        print('Please specify the path to the disaster messages file')


if __name__ == '__main__':
    main()
