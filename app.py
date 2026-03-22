from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

flower_class = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = CustomData(
            sepal_length=float(request.form.get("sepal_length")),
            sepal_width=float(request.form.get("sepal_width")),
            petal_length=float(request.form.get("petal_length")),
            petal_width=float(request.form.get("petal_width")),
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        print(result)

        return render_template('index.html', result=flower_class[int(result[0])]), 200
    except (ValueError, TypeError) as e:
        return render_template(
            'index.html',
            result="Invalid input! Please enter valide numeric values."
        ), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)