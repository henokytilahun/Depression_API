# Depression Prediction API

A production-grade REST API for predicting the probability of depression using a TensorFlow model. This service uses SHAP-based feature selection to reduce model inputs to the top 5 features, containerizes the application with Docker, and can be deployed on Render (or any Docker-compatible platform) for reliable inference.

---

## Features

* SHAP-driven feature selection (top 5 features)
* TensorFlow model for binary depression classification
* FastAPI endpoints with Pydantic validation
* Dockerfile for containerization
* Example `render.yaml` for easy deployment on Render
* Health check at `/docs` (Swagger UI)

---

## Tech Stack

* Python 3.9+
* FastAPI
* TensorFlow
* Scikit-learn (for preprocessing)
* SHAP (for interpretability)
* Docker
* Render (for cloud deployment)

---

## Repository Structure

```
├── Dockerfile
├── render.yaml
├── requirements.txt
├── main.py            # FastAPI application
├── model.py           # Training & feature-selection script
├── selected_features.json
├── model.h5
├── scaler.pkl
├── le_degree.pkl
├── le_city.pkl
├── le_gender.pkl
├── le_dietary_habits.pkl
├── le_sleep_duration.pkl
├── le_suicidal_thoughts.pkl
├── le_family_history.pkl
└── README.md
```

---

## Requirements

* Docker (for containerization)
* Python 3.9 (for local development)
* Internet connection (for deployment)

---

## Installation (Local)

1. **Clone the repository**:

   ```bash
   git clone https://github.com/henokytilahun/depression-api.git
   cd depression-api
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API**:

   ```bash
   uvicorn main:app --reload
   ```

5. **Open the docs**:
   Navigate to `http://127.0.0.1:8000/docs` for interactive Swagger UI.

---

## Usage Examples

### POST /predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "suicidal_thoughts": false,
    "academic_pressure": 3.0,
    "financial_stress": 2.5,
    "dietary_habits": "Moderate",
    "study_satisfaction": 4.0
}'
```

**Response**:

```json
{
  "depression_probability": 0.0092,
  "depression_prediction": false
}
```

---

## Docker

1. **Build the image**:

   ```bash
   docker build -t depression-api .
   ```

2. **Run the container**:

   ```bash
   docker run -d --name depression-api -p 8000:8000 depression-api
   ```

3. **Access**:

   * Swagger UI: `http://localhost:8000/docs`
   * Predict endpoint: `http://localhost:8000/predict`

---

## Deployment on Render

1. Ensure `render.yaml` is in the repo root.
2. Connect your GitHub repo to Render.
3. Render will automatically build and deploy your service on push to `main`.
4. Health check is configured at `/docs` to keep the service awake.

---

## CI/CD: Keep Awake Workflow

A GitHub Actions workflow (`.github/workflows/keep-awake.yml`) pings the `/docs` endpoint every 10 minutes to prevent the free Render service from idling.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## License

MIT License
