import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath, merge_on="id"):
    """
    Loads data from two csv files and merges them into a single
    pandas DataFrame using the merge_on column.

    Args:
        messages_filepath (str): Path to the csv file with the features.
        categories_filepath (str): Path to the csv file with the responses.
        merge_on (str, optional): Name of the column where the DataFrames will
            be merged. Defaults to "id".

    Returns:
        Pandas DataFrame: Merged DataFrame containing the features as well as
            the responses.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on=merge_on)


def clean_data(df):
    """
    Cleans a DataFrame by removing duplicates, renaming the columns and
    converting the responses into int values.Cleans a DataFrame by removing
    duplicates, renaming the columns and converting the responses into int
    values. The function also adds a "unrelated" label, for all the data that
    doesn't fit any category.

    Args:
        df (Pandas DataFrame): DataFrame with Figure Eight Disaster Data.

    Returns:
        Pandas DataFrame: Cleaned Data.
    """
    # Split the respnse column into the categories
    categories = df["categories"].str.split(";", expand=True)
    category_colnames = categories.loc[0, :].apply(lambda x: x[0:-2])
    categories.columns = category_colnames

    # Convert the categories into numeric values
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)
    df.drop("categories", axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Create a new column for any data that doesn't have a label
    df["unrelated"] = (df.iloc[:, 4:].sum(axis=1) == 0).astype(int)
    # Remove rows with the related label 2 (not translated or unfinished)
    df = df[df["related"] != 2]
    return df


def save_data(df, database_filename):
    """
    Saves the data in a SQL database.

    Args:
        df (Pandas DataFrame): DataFrame containing the Data to be stored in a
            SQL database.
        database_filename (str): Filename of the SQL database.
    """
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql(database_filename[:-3], engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath =\
            sys.argv[1:]

        print(f"""Loading data...\nMESSAGES: {messages_filepath}\n
              CATEGORIES: {categories_filepath}""")
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print(f"Saving data...\n    DATABASE: {database_filepath}")
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print("Please provide the filepaths of the messages and categories "
              "datasets as the first and second argument respectively, as "
              "well as the filepath of the database to save the cleaned data "
              "to as the third argument. \n\nExample: python process_data.py "
              "disaster_messages.csv disaster_categories.csv "
              "DisasterResponse.db")


if __name__ == "__main__":
    main()
