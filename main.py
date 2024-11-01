import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

from classes.DataProcessor import *
from classes.DatabaseHandler import *
from classes.Model import Model

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

DB_FILE = 'housing_data.db'
PROCESSED_TABLE = 'processed_data'
PREDICTIONS_TRAIN_TABLE = 'predictions_train'
PREDICTIONS_TABLE = 'predictions'
TRAIN_DATA = 'housing.csv'
MODEL_NAME = 'model.joblib'
RANDOM_STATE=100

ORIGINAL_COLUMNS = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value',
 'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
 'ocean_proximity_NEAR OCEAN']


def prepare_data(input_data_path):
    df = pd.read_csv(input_data_path)
    df = df.dropna()

    # encode the categorical variables
    df = pd.get_dummies(df)

    df_features = df.drop(['median_house_value'], axis=1)
    y = df['median_house_value'].values

    X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=RANDOM_STATE)

    return (X_train, X_test, y_train, y_test)

def train(X_train, y_train):
    # what columns are expected by the model
    X_train.columns

    regr = RandomForestRegressor(max_depth=12)
    regr.fit(X_train,y_train)

    return regr

def predict(X, model):
    Y = model.predict(X)
    return Y

def save_model(model, filename):
    with open(filename, 'wb'):
        joblib.dump(model, filename, compress=3)

def load_model(filename):
    model = joblib.load(filename)
    return model


def my_pipeline():
    """
    This function undergoes the whole pipeline. In this function, I call my custom three classes and its methods to
    successfully transform, format, and provide the input data to the model and generate predictions.

    :return: void
    """

    # Process the data
    try:
        logging.info('Processing the data...')

        processor = DataProcessor(TRAIN_DATA, ORIGINAL_COLUMNS)
        raw_data = processor.load_data()
        processed_data = processor.process_data(raw_data)

        logging.info('Data processed.')
    except Exception as e:
        logging.error(f"Error during data preparation: {e}")
        exit(1)

    # Save processed data
    logging.info('Saving processed data...')
    db_handler = DatabaseHandler(DB_FILE)
    db_handler.save_to_db(processed_data, table_name="processed_data")

    # Prepare data
    logging.info('Preparing the data...')
    data = db_handler.load_from_db("processed_data")

    # Split and format the data
    x = data.drop(['median_house_value'], axis=1)
    y = data['median_house_value'].values

    if y is None:
        print("Target column 'Price' not found in data.")
        return
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
    logging.info('Data prepared...')


    # Load the model
    logging.info('Loading the model...')
    my_model = Model(MODEL_NAME)
    logging.info('Model loaded.')


    # Calculate the predictions
    logging.info('Calculating predictions...')
    my_model.train_model(x_train, y_train)
    predictions = my_model.predict(x_test)


    # Save predictions
    logging.info('Saving predictions...')
    predictions_df = pd.DataFrame({
        'Predictions': predictions,
        'Actual': pd.Series(y_test).reset_index(drop=True)
    })
    db_handler.save_to_db(predictions_df, "predictions")

    # Display final results
    test_pred = pd.DataFrame(predictions)
    test_error = mean_absolute_error(y_test, test_pred)
    logging.info(f'Result comparison: {predictions_df.head()}')
    logging.info(f'Test error: {test_error}')
    processor.display_output(x_test, test_pred)

    db_handler.close_connection()
    logging.info('Pipeline completed successfully.')


if __name__ == '__main__':

    my_pipeline()

    """
    
    logging.info('Preparing the data...')
    X_train, X_test, y_train, y_test = prepare_data(TRAIN_DATA)

    # the model was already trained before
    # logging.info('Training the model...')
    # regr = train(TRAIN_DATA)

    # the model was already saved before into file 'model.joblib'
    # logging.info('Exporting the model...')
    # save_model(regr, MODEL_NAME)

    logging.info('Loading the model...')
    model = load_model(MODEL_NAME)

    logging.info('Calculating train dataset predictions...')
    y_pred_train = predict(X_train, model)
    logging.info('Calculating test dataset predictions...')
    y_pred_test = predict(X_test, model)

    # evaluate model
    logging.info('Evaluating the model...')
    train_error = mean_absolute_error(y_train, y_pred_train)
    test_error = mean_absolute_error(y_test, y_pred_test)

    logging.info('First 5 predictions:')
    logging.info(f'\n{X_test.head()}')
    logging.info(y_pred_test[:5])
    logging.info(f'Train error: {train_error}')
    logging.info(f'Test error: {test_error}')

    
    """
