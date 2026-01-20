import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("insurance_final_model.pkl", "rb") as f:
    model = pickle.load(f)

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def predict_charges(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([[
        age, sex, bmi, children, smoker, region
    ]], columns=["age", "sex", "bmi", "children", "smoker", "region"])

    input_df["is_obese"] = (input_df["bmi"] >= 30).astype(int)

    prediction = model.predict(input_df)[0]

    return (
        f"Predicted Medical Insurance Charges: ${prediction:,.2f}\n"
        f"BMI Category: {bmi_category(bmi)}\n"
     
    )

inputs = [
    gr.Slider(18, 64, step=1, label="Age", value=30),
    gr.Radio(["male", "female"], label="Sex", value="male"),
    gr.Slider(10.0, 60.0, step=0.1, label="BMI", value=25.0),
    gr.Slider(0, 5, step=1, label="Number of Children", value=0),
    gr.Radio(["yes", "no"], label="Smoker", value="no"),
    gr.Dropdown(
        ["southwest", "southeast", "northwest", "northeast"],
        label="Region",
        value="southeast"
    )
]

examples = [
    [25, "female", 22.5, 0, "no", "northwest"],
    [45, "male", 31.2, 2, "yes", "southeast"],
    [60, "female", 28.0, 3, "no", "southwest"]
]

app = gr.Interface(
    fn=predict_charges,
    inputs=inputs,
    outputs=gr.Textbox(lines=2, label="Prediction"),
    title="Medical Insurance Cost Predictor",
    description="Predict estimated annual medical insurance charges (USD) using a trained Gradient Boosting model.",
    examples=examples
)

app.launch(share=True)
