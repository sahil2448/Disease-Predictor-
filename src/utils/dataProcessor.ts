import { HeartDiseaseInput } from '../types/prediction';

export const processFormData = (formData: HeartDiseaseInput) => {
  // Convert form data to the format expected by the ML model
  const processedData = {
    age: formData.age,
    trestbps: formData.trestbps,
    chol: formData.chol,
    thalch: formData.thalch,
    oldpeak: formData.oldpeak,
    ca: formData.ca,
    
    // Convert boolean fields
    fbs: formData.fbs ? 1 : 0,
    exang: formData.exang ? 1 : 0,
    
    // One-hot encode categorical fields
    sex_Female: formData.sex === 'Female' ? 1 : 0,
    sex_Male: formData.sex === 'Male' ? 1 : 0,
    
    cp_asymptomatic: formData.cp === 'asymptomatic' ? 1 : 0,
    'cp_atypical angina': formData.cp === 'atypical angina' ? 1 : 0,
    'cp_non-anginal': formData.cp === 'non-anginal' ? 1 : 0,
    'cp_typical angina': formData.cp === 'typical angina' ? 1 : 0,
    
    'restecg_lv hypertrophy': formData.restecg === 'lv hypertrophy' ? 1 : 0,
    restecg_normal: formData.restecg === 'normal' ? 1 : 0,
    'restecg_st-t abnormality': formData.restecg === 'st-t abnormality' ? 1 : 0,
    
    slope_downsloping: formData.slope === 'downsloping' ? 1 : 0,
    slope_flat: formData.slope === 'flat' ? 1 : 0,
    slope_upsloping: formData.slope === 'upsloping' ? 1 : 0,
    
    'thal_fixed defect': formData.thal === 'fixed defect' ? 1 : 0,
    thal_normal: formData.thal === 'normal' ? 1 : 0,
    'thal_reversable defect': formData.thal === 'reversable defect' ? 1 : 0,
  };

  return processedData;
};