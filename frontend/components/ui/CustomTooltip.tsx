import React, { useState, useRef, useEffect } from 'react';
import { srOnly } from '@/utils/accessibilityUtils';

interface TooltipProps {
  /**
   * The content to display inside the tooltip
   */
  content: React.ReactNode;
  
  /**
   * The element that triggers the tooltip
   */
  children: React.ReactElement;
  
  /**
   * The position of the tooltip relative to the trigger element
   */
  position?: 'top' | 'right' | 'bottom' | 'left';
  
  /**
   * Delay in milliseconds before showing the tooltip
   */
  delay?: number;
  
  /**
   * Additional CSS classes to apply to the tooltip
   */
  className?: string;
  
  /**
   * Optional ID for the tooltip. If not provided, a random one will be generated
   */
  id?: string;
  
  /**
   * Whether to render tooltip content for screen readers even when not visible
   */
  alwaysRenderScreenReaderContent?: boolean;
}

/**
 * A tooltip component that displays additional information when hovering over or clicking an element.
 * Provides context and guidance without cluttering the interface.
 * Accessible to screen readers and keyboard navigation.
 */
const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  position = 'top',
  delay = 0,
  className = '',
  id,
  alwaysRenderScreenReaderContent = true,
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [tooltipStyle, setTooltipStyle] = useState({});
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Generate a unique ID for the tooltip if not provided
  const [tooltipId] = useState(() => id || `tooltip-${Math.random().toString(36).substring(2, 9)}`);

  // Show tooltip with delay
  const showTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
      calculatePosition();
    }, delay);
  };
  
  // Toggle tooltip visibility on click
  const toggleTooltip = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent triggering parent click events
    setIsVisible(!isVisible);
    if (!isVisible) {
      calculatePosition();
    }
  };
  
  // Handle keyboard events (Escape to dismiss)
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && isVisible) {
      hideTooltip();
    }
  };

  // Hide tooltip and clear any pending timeouts
  const hideTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setIsVisible(false);
  };

  // Calculate tooltip position based on trigger element and adjust for viewport boundaries
  const calculatePosition = () => {
    if (!triggerRef.current || !tooltipRef.current) return;
    
    const triggerRect = triggerRef.current.getBoundingClientRect();
    const tooltipRect = tooltipRef.current.getBoundingClientRect();
    
    // Default offset (spacing between tooltip and trigger)
    const offset = 8;
    
    // Get viewport dimensions
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Initialize position
    let style: React.CSSProperties = {};
    
    // Calculate base position based on the specified position prop
    switch (position) {
      case 'top':
        style = {
          top: `${-tooltipRect.height - offset}px`,
          left: `${(triggerRect.width - tooltipRect.width) / 2}px`,
        };
        break;
      case 'right':
        style = {
          top: `${(triggerRect.height - tooltipRect.height) / 2}px`,
          left: `${triggerRect.width + offset}px`,
        };
        break;
      case 'bottom':
        style = {
          top: `${triggerRect.height + offset}px`,
          left: `${(triggerRect.width - tooltipRect.width) / 2}px`,
        };
        break;
      case 'left':
        style = {
          top: `${(triggerRect.height - tooltipRect.height) / 2}px`,
          left: `${-tooltipRect.width - offset}px`,
        };
        break;
    }
    
    // Check if tooltip would extend beyond viewport and adjust if needed
    const tooltipAbsLeft = triggerRect.left + parseFloat(String(style.left || 0));
    const tooltipAbsTop = triggerRect.top + parseFloat(String(style.top || 0));
    
    // Adjust horizontal position
    if (tooltipAbsLeft < 0) {
      style.left = `${parseFloat(String(style.left || 0)) - tooltipAbsLeft + 8}px`;
    } else if (tooltipAbsLeft + tooltipRect.width > viewportWidth) {
      const overflow = tooltipAbsLeft + tooltipRect.width - viewportWidth;
      style.left = `${parseFloat(String(style.left || 0)) - overflow - 8}px`;
    }
    
    // Adjust vertical position
    if (tooltipAbsTop < 0) {
      style.top = `${parseFloat(String(style.top || 0)) - tooltipAbsTop + 8}px`;
    } else if (tooltipAbsTop + tooltipRect.height > viewportHeight) {
      const overflow = tooltipAbsTop + tooltipRect.height - viewportHeight;
      style.top = `${parseFloat(String(style.top || 0)) - overflow - 8}px`;
    }
    
    setTooltipStyle(style);
  };

  // Recalculate position when window is resized
  useEffect(() => {
    if (isVisible) {
      const handleResize = () => {
        calculatePosition();
      };
      
      window.addEventListener('resize', handleResize);
      return () => {
        window.removeEventListener('resize', handleResize);
      };
    }
  }, [isVisible]);

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  // Position classes based on the position prop
  const positionClasses = {
    top: 'tooltip-top',
    right: 'tooltip-right',
    bottom: 'tooltip-bottom',
    left: 'tooltip-left',
  };

  // Return early if content is empty
  if (!content) {
    return children;
  }

  return (
    <div 
      className="inline-block relative"
      ref={triggerRef}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
      onKeyDown={handleKeyDown}
      onClick={toggleTooltip} // Add click handler to toggle tooltip
    >
      {/* Clone child element with appropriate ARIA attributes */}
      {React.cloneElement(children, {
        'aria-describedby': tooltipId,
        tabIndex: children.props.tabIndex ?? (children.props.onClick ? 0 : undefined),
        onClick: (e: React.MouseEvent) => {
          // Prevent propagation and call the original onClick if it exists
          e.stopPropagation();
          toggleTooltip(e);
          if (children.props.onClick) {
            children.props.onClick(e);
          }
        }
      })}
      
      {/* Hidden content for screen readers (always available) */}
      {alwaysRenderScreenReaderContent && !isVisible && (
        <div id={tooltipId} className={srOnly}>
          {typeof content === 'string' ? content : 'Tooltip content'}
        </div>
      )}
      
      {/* Visible tooltip */}
      {isVisible && (
        <div
          id={tooltipId}
          role="tooltip"
          ref={tooltipRef}
          className={`
            absolute z-50 py-2 px-3 text-xs font-medium
            bg-gray-900 text-white rounded shadow-sm
            max-w-xs overflow-auto max-h-96
            ${positionClasses[position]} ${className}
          `}
          style={tooltipStyle}
          aria-hidden={!isVisible}
          onClick={(e) => e.stopPropagation()} // Prevent clicks inside tooltip from closing it
        >
          {content}
          <div className={`tooltip-arrow ${positionClasses[position]}`} aria-hidden="true" />
        </div>
      )}
    </div>
  );
};

export default Tooltip;