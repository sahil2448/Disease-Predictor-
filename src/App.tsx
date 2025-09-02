import React, { useState } from 'react';
import { Heart, Brain, Shield } from 'lucide-react';
import { PredictionForm } from './components/PredictionForm';
import { PredictionResultComponent } from './components/PredictionResult';
import { LoadingSpinner } from './components/LoadingSpinner';
import { HeartDiseaseInput, PredictionResult } from './types/prediction';
import { predictHeartDisease } from './services/predictionService';

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePrediction = async (data: HeartDiseaseInput) => {
    setLoading(true);
    setError(null);
    
    try {
      const predictionResult = await predictHeartDisease(data);
      setResult(predictionResult);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 animate-fade-in">
          <div className="flex justify-center mb-4">
            <div className="bg-primary-100 p-4 rounded-full">
              <Heart className="w-12 h-12 text-primary-600" />
            </div>
          </div>
          
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Heart Disease Predictor
          </h1>
          <p className="text-xl text-gray-600 mb-6">
            AI-powered cardiovascular risk assessment
          </p>
          
          <div className="flex justify-center space-x-8 text-sm text-gray-500">
            <div className="flex items-center space-x-2">
              <Brain className="w-4 h-4" />
              <span>Machine Learning</span>
            </div>
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4" />
              <span>Secure & Private</span>
            </div>
            <div className="flex items-center space-x-2">
              <Heart className="w-4 h-4" />
              <span>Clinical Grade</span>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="animate-slide-up">
          {error && (
            <div className="card bg-error-50 border-error-200 mb-6">
              <div className="flex items-center space-x-3">
                <Heart className="w-5 h-5 text-error-600" />
                <div className="text-error-800">
                  <strong>Error:</strong> {error}
                </div>
              </div>
            </div>
          )}

          {loading ? (
            <LoadingSpinner />
          ) : result ? (
            <PredictionResultComponent result={result} onReset={handleReset} />
          ) : (
            <PredictionForm onSubmit={handlePrediction} loading={loading} />
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-gray-500 text-sm">
          <p>Built with advanced machine learning algorithms trained on clinical data</p>
        </div>
      </div>
    </div>
  );
}

export default App;