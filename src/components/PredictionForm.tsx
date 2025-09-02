import React, { useState } from 'react';
import { Heart, User, Activity, Stethoscope } from 'lucide-react';
import { FormField } from './FormField';
import { HeartDiseaseInput, FormErrors } from '../types/prediction';
import { validateForm } from '../utils/validation';

interface PredictionFormProps {
  onSubmit: (data: HeartDiseaseInput) => void;
  loading: boolean;
}

export const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState<Partial<HeartDiseaseInput>>({
    age: undefined,
    trestbps: undefined,
    chol: undefined,
    fbs: false,
    restecg: '',
    thalch: undefined,
    exang: false,
    oldpeak: undefined,
    slope: '',
    ca: undefined,
    thal: '',
    sex: '',
    cp: '',
  });

  const [errors, setErrors] = useState<FormErrors>({});

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const validationErrors = validateForm(formData);
    
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    setErrors({});
    onSubmit(formData as HeartDiseaseInput);
  };

  const handleInputChange = (field: keyof HeartDiseaseInput, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Personal Information */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <User className="w-5 h-5 text-primary-600" />
          <h3 className="text-lg font-semibold text-gray-800">Personal Information</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FormField label="Age" error={errors.age} required>
            <input
              type="number"
              className="input-field"
              value={formData.age || ''}
              onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
              placeholder="Enter age"
              min="1"
              max="120"
            />
          </FormField>

          <FormField label="Sex" error={errors.sex} required>
            <select
              className="select-field"
              value={formData.sex || ''}
              onChange={(e) => handleInputChange('sex', e.target.value)}
            >
              <option value="">Select sex</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </FormField>
        </div>
      </div>

      {/* Vital Signs */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Activity className="w-5 h-5 text-primary-600" />
          <h3 className="text-lg font-semibold text-gray-800">Vital Signs</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FormField 
            label="Resting Blood Pressure" 
            error={errors.trestbps} 
            required
            description="mmHg (normal: 120/80)"
          >
            <input
              type="number"
              className="input-field"
              value={formData.trestbps || ''}
              onChange={(e) => handleInputChange('trestbps', parseInt(e.target.value))}
              placeholder="e.g., 120"
              min="80"
              max="250"
            />
          </FormField>

          <FormField 
            label="Cholesterol Level" 
            error={errors.chol} 
            required
            description="mg/dl (normal: <200)"
          >
            <input
              type="number"
              className="input-field"
              value={formData.chol || ''}
              onChange={(e) => handleInputChange('chol', parseInt(e.target.value))}
              placeholder="e.g., 200"
              min="100"
              max="600"
            />
          </FormField>

          <FormField 
            label="Maximum Heart Rate" 
            error={errors.thalch} 
            required
            description="bpm achieved during exercise"
          >
            <input
              type="number"
              className="input-field"
              value={formData.thalch || ''}
              onChange={(e) => handleInputChange('thalch', parseInt(e.target.value))}
              placeholder="e.g., 150"
              min="60"
              max="220"
            />
          </FormField>

          <FormField 
            label="ST Depression" 
            error={errors.oldpeak} 
            required
            description="Exercise induced (0-10)"
          >
            <input
              type="number"
              step="0.1"
              className="input-field"
              value={formData.oldpeak || ''}
              onChange={(e) => handleInputChange('oldpeak', parseFloat(e.target.value))}
              placeholder="e.g., 1.0"
              min="0"
              max="10"
            />
          </FormField>
        </div>
      </div>

      {/* Clinical Information */}
      <div className="card">
        <div className="flex items-center space-x-2 mb-4">
          <Stethoscope className="w-5 h-5 text-primary-600" />
          <h3 className="text-lg font-semibold text-gray-800">Clinical Information</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <FormField label="Chest Pain Type" error={errors.cp} required>
            <select
              className="select-field"
              value={formData.cp || ''}
              onChange={(e) => handleInputChange('cp', e.target.value)}
            >
              <option value="">Select chest pain type</option>
              <option value="typical angina">Typical Angina</option>
              <option value="atypical angina">Atypical Angina</option>
              <option value="non-anginal">Non-Anginal Pain</option>
              <option value="asymptomatic">Asymptomatic</option>
            </select>
          </FormField>

          <FormField label="Resting ECG" error={errors.restecg} required>
            <select
              className="select-field"
              value={formData.restecg || ''}
              onChange={(e) => handleInputChange('restecg', e.target.value)}
            >
              <option value="">Select ECG result</option>
              <option value="normal">Normal</option>
              <option value="st-t abnormality">ST-T Abnormality</option>
              <option value="lv hypertrophy">LV Hypertrophy</option>
            </select>
          </FormField>

          <FormField label="ST Slope" error={errors.slope} required>
            <select
              className="select-field"
              value={formData.slope || ''}
              onChange={(e) => handleInputChange('slope', e.target.value)}
            >
              <option value="">Select ST slope</option>
              <option value="upsloping">Upsloping</option>
              <option value="flat">Flat</option>
              <option value="downsloping">Downsloping</option>
            </select>
          </FormField>

          <FormField label="Thalassemia" error={errors.thal} required>
            <select
              className="select-field"
              value={formData.thal || ''}
              onChange={(e) => handleInputChange('thal', e.target.value)}
            >
              <option value="">Select thalassemia type</option>
              <option value="normal">Normal</option>
              <option value="fixed defect">Fixed Defect</option>
              <option value="reversable defect">Reversible Defect</option>
            </select>
          </FormField>

          <FormField 
            label="Major Vessels" 
            error={errors.ca} 
            required
            description="Number colored by fluoroscopy (0-4)"
          >
            <select
              className="select-field"
              value={formData.ca ?? ''}
              onChange={(e) => handleInputChange('ca', parseInt(e.target.value))}
            >
              <option value="">Select number</option>
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
            </select>
          </FormField>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <FormField label="Fasting Blood Sugar > 120 mg/dl">
            <div className="flex items-center space-x-4">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="fbs"
                  checked={formData.fbs === false}
                  onChange={() => handleInputChange('fbs', false)}
                  className="text-primary-600 focus:ring-primary-500"
                />
                <span>No</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="fbs"
                  checked={formData.fbs === true}
                  onChange={() => handleInputChange('fbs', true)}
                  className="text-primary-600 focus:ring-primary-500"
                />
                <span>Yes</span>
              </label>
            </div>
          </FormField>

          <FormField label="Exercise Induced Angina">
            <div className="flex items-center space-x-4">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="exang"
                  checked={formData.exang === false}
                  onChange={() => handleInputChange('exang', false)}
                  className="text-primary-600 focus:ring-primary-500"
                />
                <span>No</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="exang"
                  checked={formData.exang === true}
                  onChange={() => handleInputChange('exang', true)}
                  className="text-primary-600 focus:ring-primary-500"
                />
                <span>Yes</span>
              </label>
            </div>
          </FormField>
        </div>
      </div>

      <div className="flex justify-center">
        <button
          type="submit"
          disabled={loading}
          className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
        >
          <Heart className="w-5 h-5" />
          <span>{loading ? 'Analyzing...' : 'Predict Heart Disease Risk'}</span>
        </button>
      </div>
    </form>
  );
};