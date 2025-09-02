import { HeartDiseaseInput, FormErrors } from '../types/prediction';

export const validateForm = (data: Partial<HeartDiseaseInput>): FormErrors => {
  const errors: FormErrors = {};

  if (!data.age || data.age < 1 || data.age > 120) {
    errors.age = 'Age must be between 1 and 120';
  }

  if (!data.trestbps || data.trestbps < 80 || data.trestbps > 250) {
    errors.trestbps = 'Resting blood pressure must be between 80 and 250 mmHg';
  }

  if (!data.chol || data.chol < 100 || data.chol > 600) {
    errors.chol = 'Cholesterol must be between 100 and 600 mg/dl';
  }

  if (!data.thalch || data.thalch < 60 || data.thalch > 220) {
    errors.thalch = 'Maximum heart rate must be between 60 and 220 bpm';
  }

  if (data.oldpeak === undefined || data.oldpeak < 0 || data.oldpeak > 10) {
    errors.oldpeak = 'ST depression must be between 0 and 10';
  }

  if (data.ca === undefined || data.ca < 0 || data.ca > 4) {
    errors.ca = 'Number of major vessels must be between 0 and 4';
  }

  if (!data.sex) {
    errors.sex = 'Please select sex';
  }

  if (!data.cp) {
    errors.cp = 'Please select chest pain type';
  }

  if (!data.restecg) {
    errors.restecg = 'Please select resting ECG result';
  }

  if (!data.slope) {
    errors.slope = 'Please select ST slope';
  }

  if (!data.thal) {
    errors.thal = 'Please select thalassemia type';
  }

  return errors;
};