import logging

import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, csv_file, original_columns):
        self.csv_file = csv_file
        self.original_columns = original_columns

    def load_data(self):
        """
            Load input data from a CSV file.

            :return:
                pd.DataFrame: DataFrame containing the loaded data.
        """
        try:
            data = pd.read_csv(self.csv_file)
            return data
        except Exception as e:
            logging.info(f"Failed to read CSV file '{self.csv_file}': {e}")
            raise e

    def process_data(self, df, drop_nan=True):
        """
            Method processes the given data, fixes the known typos and enumerates the data so their are in a format suitable for the model.

            :return:
                pd.DataFrame: DataFrame containing the processed data.
        """
        if drop_nan:
            df.replace('Null', np.nan, inplace=True)
            df = df.dropna()
        else:
            df.replace('Null', 0, inplace=True)

        df.columns = df.columns.str.lower()

        # fix known typos in col names and values
        df = df.rename(columns={
            'lat': 'latitude',
            'median_age': 'housing_median_age',
            'rooms': 'total_rooms',
            'bedrooms': 'total_bedrooms',
            'pop': 'population'
        })
        df = df.replace({'ocean_proximity': {0: 'INLAND', 'OUT OF REACH': 'INLAND'}})

        # encode the categorical variables
        df = pd.get_dummies(df, columns=['ocean_proximity'])

        # make sure that the order columns in a manner how model was trained
        df, extra_cols = self.sort_columns(df)

        # make data format suitable for the model
        try:
            while len(df.columns) > len(self.original_columns):
                if extra_cols:

                    # remove columns if there are some extra (starting from last one)
                    label = extra_cols.pop(-1)
                    df.drop(label, axis=1, inplace=True)
                else:
                    logging.info('Wrong data format: Number of columns does not match with the expected format')
                    raise ValueError("Wrong data format: Number of columns does not match with the expected format")

            if len(df.columns) < len(self.original_columns):
                logging.info('Wrong data format: Number of columns does not match with the expected format')
                raise ValueError("Wrong data format: Number of columns does not match with the expected format")

        except ValueError as e:
            logging.error(f"Error adjusting columns: {e}")
            raise e

        # make all data numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        # remove outlier values
        df = self.remove_outliers(df)

        logging.info("Data processed successfully.")
        return df

    def remove_outliers(self, df):
        """
            This method removes all outlier values.

            :return:
                pd.DataFrame: DataFrame with cleaned data.
        """
        for col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def sort_columns(self, df):
        """
            Method created for sorting the columns in the right order. The expected order of the model should be kept.

            :return:
                pd.DataFrame: sorted DataFrame.
                list[]: list of unexpected columns in dataframe.
        """

        df_columns = df.columns.tolist()

        new_column_list = []
        missing_columns = []  # store columns that are missing from the expected ones
        extra_columns = list(set(df.columns) - set(self.original_columns))

        extra_columns_copy = extra_columns[:]
        df_columns = [x for x in df_columns if x not in extra_columns]

        for ori_i in range(len(self.original_columns)):
            appended = False
            for col_i in range(len(df_columns)):

                if self.original_columns[ori_i] == df_columns[col_i]:
                    new_column_list.append(df_columns[col_i])
                    appended = True
                    break

            if not appended:
                missing_columns.append(self.original_columns[ori_i])
                new_column_list.append(extra_columns_copy.pop(0))

        # join extra columns to the end of the list
        new_column_list += extra_columns_copy
        df = df.reindex(columns=new_column_list)

        if missing_columns:
            # print("Warning: These columns may be missing from expected input: + " + str(missing_columns))
            raise Warning("These columns may be missing from expected input: + " + str(missing_columns))

        if extra_columns:
            print("Unexpected columns: " + str(extra_columns))

        return df, extra_columns

    def display_output(self, data, predictions):
        """
            Display an example of input data and its prediction.
        """

        ocean_proximity_mapping = {
            'ocean_proximity_<1H OCEAN': '<1H OCEAN',
            'ocean_proximity_INLAND': 'INLAND',
            'ocean_proximity_ISLAND': 'ISLAND',
            'ocean_proximity_NEAR BAY': 'NEAR BAY',
            'ocean_proximity_NEAR OCEAN': 'NEAR OCEAN'
        }

        input_row = data.iloc[0]
        prediction_row = predictions.iloc[0]
        ocean_proximity_value = next((v for k, v in ocean_proximity_mapping.items() if input_row[k] == 1.0), None)

        print("Input values:")
        for col, value in input_row.items():
            if 'ocean_proximity' in col:  # skip the one we used get_dummies() on
                continue
            print(f"{col}: {value}")

        print(f"ocean_proximity: '{ocean_proximity_value}'")

        print("\nPrediction output:")
        print(f"output: {prediction_row[0]}")