import React from 'react';
import { AlertCircle } from 'lucide-react';

interface FormFieldProps {
  label: string;
  error?: string;
  required?: boolean;
  children: React.ReactNode;
  description?: string;
}

export const FormField: React.FC<FormFieldProps> = ({
  label,
  error,
  required = false,
  children,
  description
}) => {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-gray-700">
        {label}
        {required && <span className="text-error-500 ml-1">*</span>}
      </label>
      {description && (
        <p className="text-xs text-gray-500">{description}</p>
      )}
      {children}
      {error && (
        <div className="flex items-center space-x-2 text-error-600 text-sm animate-slide-up">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};