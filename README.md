# Heart Disease Predictor

An AI-powered web application for cardiovascular risk assessment using machine learning algorithms trained on clinical data.

## Features

- **Interactive Web Interface**: Modern, responsive design with intuitive form inputs
- **Real-time Predictions**: Instant heart disease risk assessment
- **Clinical-grade Analysis**: Based on established cardiovascular risk factors
- **Risk Stratification**: Clear low/moderate/high risk categorization
- **Medical Recommendations**: Actionable advice based on risk level
- **Secure & Private**: All data processing happens locally

## Technology Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Vite** for fast development and building
- **Lucide React** for icons

### Backend
- **Flask** Python web framework
- **scikit-learn** for machine learning
- **pandas** for data processing
- **joblib** for model serialization

## Getting Started

### Prerequisites
- Node.js 18+ 
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Install frontend dependencies:**
   ```bash
   npm install
   ```

2. **Install backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

### Development

1. **Start both frontend and backend:**
   ```bash
   npm start
   ```

   This will start:
   - Frontend development server on `http://localhost:5173`
   - Backend API server on `http://localhost:5000`

2. **Or start them separately:**
   
   Frontend only:
   ```bash
   npm run dev
   ```
   
   Backend only:
   ```bash
   npm run backend
   ```

## Model Information

The heart disease prediction model uses the following features:

### Demographic
- Age
- Sex

### Clinical Measurements
- Resting blood pressure (trestbps)
- Cholesterol level (chol)
- Maximum heart rate achieved (thalch)
- ST depression induced by exercise (oldpeak)
- Number of major vessels colored by fluoroscopy (ca)

### Clinical Indicators
- Chest pain type (cp)
- Fasting blood sugar > 120 mg/dl (fbs)
- Resting electrocardiographic results (restecg)
- Exercise induced angina (exang)
- Slope of peak exercise ST segment (slope)
- Thalassemia type (thal)

## API Endpoints

### `GET /health`
Health check endpoint to verify API status.

### `POST /predict`
Predict heart disease risk based on input features.

**Request Body:**
```json
{
  "age": 63,
  "sex": "Male",
  "cp": "typical angina",
  "trestbps": 145,
  "chol": 233,
  "fbs": true,
  "restecg": "normal",
  "thalch": 150,
  "exang": false,
  "oldpeak": 2.3,
  "slope": "downsloping",
  "ca": 0,
  "thal": "fixed defect"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.75,
  "risk_level": "High",
  "confidence": 0.89
}
```

## Deployment

The application is ready for deployment to various platforms:

- **Frontend**: Can be deployed to Vercel, Netlify, or any static hosting service
- **Backend**: Can be deployed to Heroku, Railway, or any Python hosting service

## Medical Disclaimer

This application is for educational and research purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for proper medical evaluation and care.

## License

This project is open source and available under the MIT License.