def test_index_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Iris" in response.data  # depends on your HTML

def test_predict_success(client, mocker):
    # Mock PredictPipeline
    mock_pipeline = mocker.patch("app.PredictPipeline")
    mock_instance = mock_pipeline.return_value
    mock_instance.predict.return_value = [0]  # setosa

    data = {
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2"
    }

    response = client.post("/predict", data=data)

    assert response.status_code == 200
    assert b"setosa" in response.data

def test_predict_invalid_input(client):
    data = {
        "sepal_length": "invalid",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2"
    }

    response = client.post("/predict", data=data)

    assert response.status_code in [400, 500]

def test_predict_missing_fields(client):
    data = {
        "sepal_length": "5.1"
    }

    response = client.post("/predict", data=data)

    assert response.status_code in [400, 500]

def test_pipeline_called(client, mocker):
    mock_pipeline = mocker.patch("app.PredictPipeline")
    mock_instance = mock_pipeline.return_value
    mock_instance.predict.return_value = [2]

    data = {
        "sepal_length": "6.5",
        "sepal_width": "3.0",
        "petal_length": "5.2",
        "petal_width": "2.0"
    }

    response = client.post("/predict", data=data)

    mock_instance.predict.assert_called_once()
    assert b"virginica" in response.data