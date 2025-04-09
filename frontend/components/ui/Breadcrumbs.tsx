import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import Tooltip from './CustomTooltip';

export interface BreadcrumbItem {
  /**
   * Label to display for this breadcrumb item
   */
  label: string;
  
  /**
   * URL to navigate to when clicked (if not provided, item is not clickable)
   */
  href?: string;
  
  /**
   * Optional icon to display before the label
   */
  icon?: React.ReactNode;
}

interface BreadcrumbsProps {
  /**
   * Array of breadcrumb items to display
   */
  items: BreadcrumbItem[];
  
  /**
   * Optional maximum number of items to show (truncates middle items if exceeded)
   */
  maxItems?: number;
  
  /**
   * Optional additional CSS classes
   */
  className?: string;
}

/**
 * A responsive breadcrumb navigation component that shows the user's current location
 * within the application hierarchy.
 */
const Breadcrumbs: React.FC<BreadcrumbsProps> = ({
  items,
  maxItems = 0,
  className = '',
}) => {
  const router = useRouter();
  
  // Handle case with no items
  if (!items || items.length === 0) {
    return null;
  }
  
  // Apply maxItems truncation if needed
  let displayedItems = [...items];
  
  if (maxItems > 0 && items.length > maxItems) {
    // Always keep first and last items
    const firstItem = items[0];
    const lastItems = items.slice(-1 * (maxItems - 1));
    
    // Add ellipsis item
    displayedItems = [
      firstItem,
      { label: '...', href: undefined },
      ...lastItems
    ];
  }
  
  return (
    <nav 
      className={`flex items-center py-2 ${className}`}
      aria-label="Breadcrumb"
    >
      <ol className="inline-flex items-center space-x-1 md:space-x-2 flex-wrap">
        {displayedItems.map((item, index) => (
          <li key={index} className="inline-flex items-center">
            {/* Separator between items (except first) */}
            {index > 0 && (
              <span className="mx-1 md:mx-2 text-gray-400" aria-hidden="true">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              </span>
            )}
            
            <div className="flex items-center">
              {/* Home icon for first item */}
              {index === 0 && !item.icon && (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-4 w-4 mr-1"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
                  />
                </svg>
              )}
              
              {/* Custom icon if provided */}
              {item.icon && (
                <span className="mr-1">{item.icon}</span>
              )}
              
              {/* Render links for items with href, plain text otherwise */}
              {item.href ? (
                <Link
                  href={item.href}
                  className={`text-sm font-medium 
                    ${index === displayedItems.length - 1
                      ? 'text-gray-600 dark:text-gray-400 cursor-default pointer-events-none'
                      : 'text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300'
                    }`}
                  aria-current={index === displayedItems.length - 1 ? 'page' : undefined}
                >
                  {item.label}
                </Link>
              ) : (
                <Tooltip
                  content={index === 1 && maxItems > 0 ? "More breadcrumbs hidden" : ""}
                  delay={500}
                >
                  <span
                    className="text-sm font-medium text-gray-600 dark:text-gray-400"
                    aria-current={index === displayedItems.length - 1 ? 'page' : undefined}
                  >
                    {item.label}
                  </span>
                </Tooltip>
              )}
            </div>
          </li>
        ))}
      </ol>
    </nav>
  );
};

/**
 * Helper function to generate breadcrumb items from a URL path
 * @param pathname The current pathname (from router.pathname)
 * @param labels Optional map of path segments to custom labels
 */
export const generateBreadcrumbs = (
  pathname: string,
  labels: Record<string, string> = {}
): BreadcrumbItem[] => {
  // Always start with home
  const breadcrumbs: BreadcrumbItem[] = [
    { label: 'Home', href: '/' }
  ];
  
  // Skip if we're already on the home page
  if (pathname === '/') {
    return breadcrumbs;
  }
  
  // Split the pathname into segments and build up paths
  const segments = pathname.split('/').filter(Boolean);
  let currentPath = '';
  
  segments.forEach((segment, index) => {
    currentPath += `/${segment}`;
    
    // Special handling for dynamic route segments (those starting with [)
    if (segment.startsWith('[') && segment.endsWith(']')) {
      // We don't add these as breadcrumbs since they're just route parameters
      return;
    }
    
    // Use custom label if available, otherwise format the segment
    const label = labels[segment] || 
      segment.charAt(0).toUpperCase() + segment.slice(1).replace(/-/g, ' ');
    
    // Last segment doesn't get a link (it's the current page)
    const isLastSegment = index === segments.length - 1;
    
    breadcrumbs.push({
      label,
      href: isLastSegment ? undefined : currentPath
    });
  });
  
  return breadcrumbs;
};

export default Breadcrumbs;
