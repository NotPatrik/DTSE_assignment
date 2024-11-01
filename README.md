
# DTSE Data Engineer (ETL) assignment Solution
## Elaborated by Patrik Minarovič


In the provided environment I created a directory `classes` with three python files each containing its own class (`DataProcessor.py`, `DatabaseHandler.py`, `Model.py`), and created a function `my_pipeline()` in the `main.py` file.
While implementing these functions, I tried to implement it in a way that should be is easier to modify in the future work.
I tied to create a general solution to the problem, but at the same time, I do believe that I did not think of every possible scenario.

I tried to keep the code clean and leave useful comments for easier read through of my solution.

Following next, here is the basic description of my classes and function:

### `mian.py`
> I declared multiple  **Global variables**, one of them is called `ORIGINAL_COLUMNS`, holding a list of columns names of the expected input for the model.
> This list I acquired from the received model under attribute `model.feature_names_in_`.
> I tried to minimize editing the `main.py` file. I did just a minor changes and tried to implement the functionality inside my function.

### `DatabaseHandler.py`
>  This class holds a constructor and three methods. Class is responsible for handling all the interaction with the database:
>> 1. `__init__(self, db_name='housing_data.db')` - Initialize the connection to the database.
>>    * `db_name` - Path / Name of the database file. 
>> 2. `save_to_db(df, db_file, table_name)` 
>>    * `df` - Expects a DataFrame
>>    * `db_file` - Path to a `.db` file (if it doesn't exist, new one will be created)
>>    * `table_name` - Name of the table where data (`df`) should be stored
>> 3. `load_from_db(db_file, table_name)`
>>    * `db_file` - Path to a `.db` file to read from.
>>    * `table_name` - Name of the table we want to read from.
>> 4. `close_connection(self)` - Close connection to the database.



### `DataProcessor.py`
> This class holds functions related to processing the data, mainly the transformation of them.
> 
>File consists of these five methods and constructor: 
>>1. `__init__(self, csv_file, original_columns)`
>>   * `csv_file` - Path to the input data (`housing.csv`).
>>   * `original_columns` - List of expected columns be the model.
>>2. `load_data(self)`
>>3. `process_data(self, df, drop_nan=True)`
>>   * `df` - DataFrame containing the Data.
>>   * `drop_na` - Refers to a mode, whether to drop the NaN values (True), or replace them with 0 (False).
>>4. `remove_outliers(self, df)`
>>   * `df` - DataFrame containing the Data.
>>5. `sort_columns(self, df)`
>>   * `df` - DataFrame containing the Data.
>>6. `display_output(self, data, predictions)`
>>   * `data` - DataFrame containing the input data for the model.
>>   * `predictions` - Output predictions generated by the model with the `data` as input.
>
> I implemented `remove_outliers()` function, based on my previous experience with data pipeline. Outlier values can negatively influence the model.


### `Model.py`
> Class covers the model interaction, such as loading the model, training and also generating predictions.
> 
> > 1. `__init__(self, model)`
> >    * `model` - Path / Name of the model to be loaded.
> > 2. `train_model(self, x_train, y_train)`
> >    * `x_train` - Data to learn on during the training process.
> >    * `y_train` - Data to validate and correct the predictions during the training process.
> > 3. `predict(self, data)`
> >    * `data` - DataFrame holding the input data to generate the predictions from.

### Output example
Here is an output example of the code.
```
INFO:root:Result comparison:      Predictions    Actual
0   99042.183545   89500.0
1  367621.428042  446800.0
2  143170.387986  133900.0
3  196544.614336  176300.0
4  248471.130166  371700.0

INFO:root:Test error: 27263.24680318065

Input values:
  longitude: -118.24
  latitude: 33.95
  housing_median_age: 40.0
  total_rooms: 1193.0
  total_bedrooms: 280.0
  population: 1210.0
  households: 286.0
  median_income: 1.35
  ocean_proximity: '<1H OCEAN'

Prediction output:
  output: 99042.18354465984

```

## Optional Tasks

- Logging
  - I tied to implement them in my solution, as a demonstration of **how** I would implement them. The point is to log when is each action before it begins. I think there also should be a log, when the action or process is finished.
  - The reason processes should be logged, is for easier debugging when something goes wrong.
  ```angular2html
    logging.info('Loading the model...')
    model = load_model(MODEL_NAME)
    logging.info('Model loaded.')
  ```
- Tests
  - Automated tests, like Unit tests for example, will make sure that our code / application is still working as we expect it to work or behave. After updating or making any changes, unintentionally we can change the behavior of our code. These tests help us to deliver a code we intend to deliver.
  - I think testing is one of the most neglected aspects of development.
  - As an example for implementation in this assignment, one of the test could check whether the input data have the expected format. Something like this:
  ```angular2html
    class TestStringMethods(unittest.TestCase):
        def test_format(self, data):
            formatted_data = format_data(data)
            self.assertEqual(formatted_data, EXPECTED_FORMAT) 
  ```
- Exception handling
  - I would implement it with try-except methods.
  - Each separate process I would put into separate try{} brackets. This way it's easier to handle unexpected inputs,...
  - I tried to implement try-except method in main.py, It may be a bit of an overkill, but this way when an error occurs, it's easier to track down its origin.
  ```angular2html
      try:
          logging.info('Preparing the data...')
          X_train, X_test, y_train, y_test = prepare_data(TRAIN_DATA)
          logging.info('Data prepared.')
      except Exception as e:
          logging.error(f"Error during data preparation")
          exit(1)
  ```
- API
  - I have some experience with handling API requests from school and also from work. In python, I have worked with `fastapi` library, so I would use that for implementing endpoints, etc.
  - As for a why, API serves as a base to connect back-end and front-end of an application. Using API requests also help with security and overall control of data. 
  - In this assigment we could use API requests in handling the data, pulling the data from the database on the server.
