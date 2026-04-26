from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
import re


spark = SparkSession.builder.appName ('Parquet').getOrCreate ()

disease_description = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Medical Diagnosis/SympScan/description.csv')
disease_medication_encouragement = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Medical Diagnosis/SympScan/medications.csv')
disease_diet_encouragement = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Medical Diagnosis/SympScan/diets.csv')
disease_workout_encouragement = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Medical Diagnosis/SympScan/workout.csv')
disease_and_precaution = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Medical Diagnosis/SympScan/precautions.csv')
disease_and_symptom = pd.read_csv ('/Users/chikhoado/Desktop/PROJECTS/Medical Diagnosis/SympScan/Diseases_and_Symptoms_dataset.csv')

disease_and_symptom_column = disease_and_symptom.columns
array_json_text = []
array_flatten_text = []


def dataset_ingestion_type_1 (row_index, dataset_name):

    if dataset_name == 'medication':
        string = disease_medication_encouragement.loc[row_index]['Medication'].strip ()

    elif dataset_name == 'diet':
        string = disease_diet_encouragement.loc[row_index]['Diet'].strip ()

    string_processed = re.split (r"'", string)
    text = ""

    for index in range (len (string_processed)):
        if index % 2 == 0:
            continue
    
        text += string_processed[index].strip ()

        if index + 2 == len (string_processed):
            text += '.'
        else:
            text += ', '
    
    return text

def dataset_ingestion_type_2 (row_index, dataset_name):

    if dataset_name == 'workout':
        string = disease_workout_encouragement.loc[row_index]['Workouts'].strip ()

    string = string[1 : len (string) - 1]
    string_processed = re.split ('"', string)
    text = ""

    for index in range (len (string_processed)):
        if index % 2 == 0:
            continue

        string_temp = string_processed[index].strip ()
        string_temp = re.split (":", string_temp)

        key = string_temp[0].strip ()
        text += key

        if len (string_temp) > 1:
            description = string_temp[1]
            description = re.split (",", description)

            text += " ("
            for idx in range (len (description)):
                text += description[idx].strip ()

                if idx + 1 < len (description):
                    text += ', '
            text += ")"

        if index + 1 == len (string_processed):
            text += '.'
        else:
            text += ', '

    return text

def dataset_ingestion_type_3 (row_index, dataset_name):
    
    text = ""
    for column_index in range (1, disease_and_precaution.shape[1]):
        column_name = f'Precaution_{column_index}'
        precaution = disease_and_precaution.iloc[row_index][column_name]

        precaution = precaution[0].lower () + precaution[1 : ]
        text += precaution

        if column_index + 1 == disease_and_precaution.shape[1]:
            text += '.'
        else:
            text += ', '

    return text

def dataset_ingestion_type_4 (disease_name, dataset_name):

    disease_name = disease_name.lower ()
    dataset_extracted = disease_and_symptom[disease_and_symptom['diseases'] == disease_name]

    patient = 'patient record #'
    text = ""
    index = 0

    for _, rows in dataset_extracted.iterrows ():
        index += 1

        true_false_coding = rows == 1
        symptoms = disease_and_symptom_column[true_false_coding]

        symptoms_text = ""
        for idx in range (len (symptoms)):
            symptoms_text += symptoms[idx]

            if idx + 1 != len (symptoms):
                symptoms_text += ', '

        patient_id = patient + str (index) + ' ('
        text += patient_id 
        text += symptoms_text
        text += ')'

        if index == dataset_extracted.shape[0]:
            text += '.'
        else:
            text += ', '

    return text

def save_processed_dataset ():

    dataset_combined = []
    for index in range (len (array_flatten_text)):
        dataset_combined.append ((array_flatten_text[index], array_json_text[index]))

    json_schema = StructType([
            StructField ("disease_name", StringType (), False),
            StructField ("disease_description", StringType (), False),
            StructField ("disease_treatment_plan", StringType (), False),
            StructField ("disease_dietary_guidelines", StringType (), False),
            StructField ("disease_exercise_protocol", StringType (), False),
            StructField ("disease_precautions", StringType (), False),
            StructField ("disease_symptom_profile", StringType (), False)
        ])
    
    dataset_schema = StructType ([
                        StructField ("flatten_dataset", StringType (), False),
                        StructField ("json_dataset", json_schema, False)
                    ])

    dataset_parquet = spark.createDataFrame (dataset_combined, schema = dataset_schema)
    dataset_parquet.write.mode ("overwrite").parquet ("Processed_Dataset.parquet")


if __name__ == '__main__':

    for row_index in disease_description.index:
        disease_name = disease_description.loc[row_index]['Disease']
        disease_description_text = disease_description.loc[row_index]['Description']

        disease_treatment_plan = dataset_ingestion_type_1 (row_index, 'medication')
        disease_dietary_guidelines = dataset_ingestion_type_1 (row_index, 'diet')
        disease_exercise_protocol = dataset_ingestion_type_2 (row_index, 'workout')
        disease_precautions = dataset_ingestion_type_3 (row_index, 'precaution')
        disease_symptom_profile = dataset_ingestion_type_4 (disease_name, 'symptom')

        json_format = {
                "disease_name": disease_name,
                "disease_description": disease_description_text,
                "disease_treatment_plan": disease_treatment_plan,
                "disease_dietary_guidelines": disease_dietary_guidelines,
                "disease_exercise_protocol": disease_exercise_protocol,
                "disease_precautions": disease_precautions,
                "disease_symptom_profile": disease_symptom_profile
            }
        
        flatten_format = (
                f"DISEASE NAME: {disease_name} "
                f"DISEASE DESCRIPTION: {disease_description_text} "
                f"DISEASE TREATMENT PLAN: {disease_treatment_plan} "
                f"DISEASE DIETARY GUIDELINES: {disease_dietary_guidelines} "
                f"DISEASE EXERCISE PROTOCOL: {disease_exercise_protocol} "
                f"DISEASE PRECAUTIONS: {disease_precautions} "
                f"DISEASE SYMPTOM PROFILE: {disease_symptom_profile}"
            )
        
        array_json_text.append (json_format)
        array_flatten_text.append (flatten_format)

    save_processed_dataset ()
    




# cd '/Users/chikhoado/Desktop/PROJECTS/Medical Diagnosis'
# /opt/homebrew/bin/python3.12 -m venv .venv
# source .venv/bin/activate
# pip install pyspark pandas regex 
# python '/Users/chikhoado/Desktop/PROJECTS/Medical Diagnosis/Raw_Dataset_PreProcess.py'





# https://www.kaggle.com/datasets/behzadhassan/sympscan-symptomps-to-disease