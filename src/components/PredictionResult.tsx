import React from 'react';
import { Heart, Shield, AlertTriangle, TrendingUp } from 'lucide-react';
import { PredictionResult } from '../types/prediction';

interface PredictionResultProps {
  result: PredictionResult;
  onReset: () => void;
}

export const PredictionResultComponent: React.FC<PredictionResultProps> = ({ result, onReset }) => {
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low': return 'text-success-600 bg-success-50 border-success-200';
      case 'moderate': return 'text-warning-600 bg-warning-50 border-warning-200';
      case 'high': return 'text-error-600 bg-error-50 border-error-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low': return <Shield className="w-8 h-8" />;
      case 'moderate': return <AlertTriangle className="w-8 h-8" />;
      case 'high': return <Heart className="w-8 h-8" />;
      default: return <TrendingUp className="w-8 h-8" />;
    }
  };

  const getRecommendations = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low':
        return [
          'Maintain a healthy lifestyle with regular exercise',
          'Continue with balanced diet and regular check-ups',
          'Monitor blood pressure and cholesterol levels annually'
        ];
      case 'moderate':
        return [
          'Consult with a cardiologist for detailed evaluation',
          'Consider lifestyle modifications (diet, exercise, stress management)',
          'Monitor cardiovascular health more frequently',
          'Discuss preventive medications with your doctor'
        ];
      case 'high':
        return [
          'Seek immediate medical attention from a cardiologist',
          'Undergo comprehensive cardiac evaluation',
          'Follow strict medical supervision and treatment plan',
          'Make immediate lifestyle changes as recommended by your doctor'
        ];
      default:
        return ['Consult with a healthcare professional for proper evaluation'];
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Main Result Card */}
      <div className={`card border-2 ${getRiskColor(result.risk_level)}`}>
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            {getRiskIcon(result.risk_level)}
          </div>
          
          <div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">
              Prediction Result
            </h2>
            <div className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-semibold ${getRiskColor(result.risk_level)}`}>
              {result.risk_level.toUpperCase()} RISK
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600">Probability</div>
              <div className="text-2xl font-bold text-gray-800">
                {(result.probability * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600">Confidence</div>
              <div className="text-2xl font-bold text-gray-800">
                {(result.confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center space-x-2">
          <TrendingUp className="w-5 h-5 text-primary-600" />
          <span>Recommendations</span>
        </h3>
        
        <ul className="space-y-3">
          {getRecommendations(result.risk_level).map((recommendation, index) => (
            <li key={index} className="flex items-start space-x-3">
              <div className="w-2 h-2 bg-primary-500 rounded-full mt-2 flex-shrink-0"></div>
              <span className="text-gray-700">{recommendation}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Disclaimer */}
      <div className="card bg-yellow-50 border-yellow-200">
        <div className="flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
          <div className="text-sm text-yellow-800">
            <strong>Medical Disclaimer:</strong> This prediction is for educational purposes only and should not replace professional medical advice. Always consult with qualified healthcare professionals for proper diagnosis and treatment.
          </div>
        </div>
      </div>

      <div className="flex justify-center">
        <button
          onClick={onReset}
          className="btn-secondary"
        >
          Make Another Prediction
        </button>
      </div>
    </div>
  );
};