import React, { ButtonHTMLAttributes, ReactNode } from 'react';

export type ButtonVariant = 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
export type ButtonSize = 'default' | 'sm' | 'lg' | 'icon';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
}

const getButtonClasses = (
  variant: ButtonVariant = 'default',
  size: ButtonSize = 'default',
  className?: string
): string => {
  // Base classes
  let classes =
    'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none';

  // Variant classes
  switch (variant) {
    case 'default':
      classes += ' bg-primary-600 text-white hover:bg-primary-700';
      break;
    case 'destructive':
      classes += ' bg-red-600 text-white hover:bg-red-700';
      break;
    case 'outline':
      classes +=
        ' border border-primary-200 dark:border-primary-800 bg-transparent hover:bg-primary-100 dark:hover:bg-primary-900';
      break;
    case 'secondary':
      classes +=
        ' bg-secondary-200 dark:bg-secondary-800 text-secondary-900 dark:text-secondary-100 hover:bg-secondary-300 dark:hover:bg-secondary-700';
      break;
    case 'ghost':
      classes +=
        ' bg-transparent hover:bg-primary-100 dark:hover:bg-primary-900 text-primary-900 dark:text-primary-100';
      break;
    case 'link':
      classes +=
        ' bg-transparent underline-offset-4 hover:underline text-primary-900 dark:text-primary-100';
      break;
  }

  // Size classes
  switch (size) {
    case 'default':
      classes += ' h-10 py-2 px-4';
      break;
    case 'sm':
      classes += ' h-8 px-3 text-xs';
      break;
    case 'lg':
      classes += ' h-12 px-6 text-base';
      break;
    case 'icon':
      classes += ' h-10 w-10';
      break;
  }

  // Add custom classes
  if (className) {
    classes += ` ${className}`;
  }

  return classes;
};

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', isLoading, children, ...props }, ref) => {
    return (
      <button
        className={getButtonClasses(variant, size, className)}
        ref={ref}
        disabled={isLoading || props.disabled}
        {...props}
      >
        {isLoading ? (
          <div className="mr-2">
            <svg
              className="animate-spin h-4 w-4 text-current"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
          </div>
        ) : null}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

export { Button };
