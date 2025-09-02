export interface HeartDiseaseInput {
  age: number;
  trestbps: number;
  chol: number;
  fbs: boolean;
  restecg: string;
  thalch: number;
  exang: boolean;
  oldpeak: number;
  slope: string;
  ca: number;
  thal: string;
  sex: string;
  cp: string;
}

export interface PredictionResult {
  prediction: number;
  probability: number;
  risk_level: string;
  confidence: number;
}

export interface FormErrors {
  [key: string]: string;
}