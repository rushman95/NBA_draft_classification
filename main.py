import os
import argparse
import pandas as pd
import joblib
import pandas as pd

 

def convert_height(height_str):

    """
    Converts a height date string to inches.

    """

    if pd.isna(height_str) or height_str in ['-', 'None', 'So', 'Jr', 'Fr', '0']:
        return None
    elif height_str == 'Apr-00':
        return 4*12
    elif height_str == 'May-00':
        return 5*12
    elif height_str == 'Jun-00':
        return 6*12
    elif height_str == 'Jul-00':
        return 7*12
    elif "'" in height_str:
        feet, inches = height_str.split("'")
        return int(feet) * 12 + int(inches)
    elif '-' in height_str:
        try:
            inch, ft = height_str.split("-")
            if ft == 'Jun':
                return int(inch) + 6*12
            elif ft == 'Jul':
                return int(inch) + 7*12
            elif ft == 'May':
                return int(inch) + 5*12
            elif ft == 'Apr':
                return int(inch) + 4*12
        except ValueError:
            return height_str
    else:
        return height_str


def load_data(input_data_location):
    """
    Load input data and convert height.
    """

    # Load the input data
    try:
        input_data = pd.read_csv(input_data_location)
        input_data['ht'] = input_data['ht'].apply(convert_height)

        return input_data
    except Exception as e:
        print(f"Error loading input data: {str(e)}")


def load_model():
    """
    Load a trained machine learning model from the given file path.
    """
    try:
        model_path = "../NBA_draft_classification/models/final.pkl"
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load the model: {str(e)}")

def save_predictions(predictions, input_data_location):
    """
    Save predictions to same location as input file.
    """
    try:
        
        directory = os.path.dirname(input_data_location)
        filename = os.path.basename(input_data_location)

        output_path = f"{directory}/predictions_{filename}"

        predictions.to_csv(output_path, index=False)

        print(f"Predictions saved to {output_path}")
    except Exception as e:
        print(f"Error saving predictions to {output_path}: {str(e)}")

def predict(model, input_data):
    """
    Make predictions using the loaded machine learning model.
    """
    try:
        # Preprocess input_data as needed (e.g., feature engineering, data cleaning)
        
        # Make predictions
        input_data['predictions'] = model.predict(input_data)
        
        return input_data
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("--input_data", type=str, required=True, help="Path to the input data file for predictions.")
    
    args = parser.parse_args()

    # Load the trained model
    model = load_model()

    # Load the input data (e.g., CSV file)
    input_data = load_data(args.input_data)

    # Make predictions
    predictions = predict(model, input_data)

    save_predictions(predictions, args.input_data)




