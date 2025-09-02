import React from 'react';
import { Heart } from 'lucide-react';

export const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center space-y-4 py-12">
      <div className="relative">
        <Heart className="w-12 h-12 text-primary-600 animate-pulse" />
        <div className="absolute inset-0 w-12 h-12 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin"></div>
      </div>
      <div className="text-center">
        <h3 className="text-lg font-semibold text-gray-800">Analyzing Your Data</h3>
        <p className="text-gray-600">Our AI model is processing your information...</p>
      </div>
    </div>
  );
};