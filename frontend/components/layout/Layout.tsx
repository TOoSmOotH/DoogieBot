import React, { ReactNode, useState, useEffect, useRef } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { useShortcuts } from '@/contexts/ShortcutContext';
import Link from 'next/link';
import { ariaLandmarks, srOnly, srOnlyFocusable, announce } from '@/utils/accessibilityUtils';
import SkipLink from '@/components/ui/SkipLink';
import ProfileDropdown from '@/components/ui/ProfileDropdown';
import ExportDropdown from '@/components/ui/ExportDropdown';
import { Input } from '@/components/ui/Input';

interface LayoutProps {
  children: ReactNode;
  title?: string;
  description?: string;
  chatHistory?: ReactNode;       // Kept for backward compatibility
  activeConversationId?: string; // Kept for backward compatibility
  sidebarContent?: ReactNode;    // New prop that replaces chatHistory
  isSidebarOpen?: boolean;       // New prop to control initial sidebar visibility
  hideDefaultSidebar?: boolean;  // New prop to hide the default sidebar completely
}

const Layout: React.FC<LayoutProps> = ({
  children,
  title = 'Doogie Chat Bot',
  description = 'A sophisticated chat bot with a Hybrid RAG system',
  chatHistory,
  activeConversationId,
  sidebarContent,
  isSidebarOpen = false,
  hideDefaultSidebar = false
}) => {
  const { isAuthenticated, isAdmin, user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const shortcuts = useShortcuts();
  const router = useRouter();
  
  // Add state for user menu
  const [showUserMenu, setShowUserMenu] = useState(false);
  
  // Title editing state
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editedTitle, setEditedTitle] = useState(title);
  const titleInputRef = useRef<HTMLInputElement>(null);
  
  // Update edited title when parent component title changes
  useEffect(() => {
    setEditedTitle(title);
  }, [title]);
  
  // Handle saving the edited title
  const handleSaveTitle = () => {
    if (!editedTitle.trim() || editedTitle === title) {
      setIsEditingTitle(false);
      return;
    }
    
    // Dispatch event to notify the ChatPage component
    if (router.pathname.startsWith('/chat') && router.query.id) {
      console.log('Dispatching edit-chat-title-completed event with chatId:', router.query.id, 'and newTitle:', editedTitle);
      const event = new CustomEvent('edit-chat-title-completed', { 
        detail: { 
          chatId: router.query.id,
          newTitle: editedTitle
        } 
      });
      document.dispatchEvent(event);
    }
    
    setIsEditingTitle(false);
  };
  
  // Focus the input when entering edit mode
  useEffect(() => {
    if (isEditingTitle && titleInputRef.current) {
      titleInputRef.current.focus();
      titleInputRef.current.select();
    }
  }, [isEditingTitle]);
  
  // Sidebar states
  const [isPinned, setIsPinned] = useState(() => {
    // Only run in client-side
    if (typeof window !== 'undefined') {
      const storedValue = localStorage.getItem('sidebarPinned');
      // If a value exists in localStorage, use that; otherwise default to pinned (true)
      return storedValue !== null ? storedValue === 'true' : true;
    }
    return true; // Default to pinned on server-side rendering
  });
  
  const [isSidebarVisible, setSidebarVisible] = useState(() => {
    // If we're on the client
    if (typeof window !== 'undefined') {
      // Check if there's a stored pin value
      const storedValue = localStorage.getItem('sidebarPinned');
      // If stored value exists, use it; if it's null, default to true (pinned)
      const isPinnedFromStorage = storedValue !== null ? storedValue === 'true' : true;
      // If pinned, always show sidebar; otherwise use the prop
      return isPinnedFromStorage ? true : isSidebarOpen;
    }
    // On server, follow the same logic - default to visible since isPinned defaults to true
    return true;
  });
  const sidebarRef = useRef<HTMLDivElement>(null);
  const mainContentRef = useRef<HTMLDivElement>(null);
  const autoHideTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Track if we're hovering over the sidebar trigger area
  const [isHoveringTrigger, setIsHoveringTrigger] = useState(false);
  
  // Determine what to show in the sidebar
  const sidebarContentToShow = sidebarContent || (
    isAuthenticated && router.pathname.startsWith('/chat') && chatHistory ? (
      <div className="mb-4">
        <h2 className="text-lg font-medium text-white mb-2">Conversations</h2>
        {chatHistory}
      </div>
    ) : null
  );
  
  // Update sidebar visibility when isSidebarOpen prop changes
  useEffect(() => {
    // Only update isSidebarVisible from props if not pinned
    if (!isPinned) {
      setSidebarVisible(isSidebarOpen);
    }
  }, [isSidebarOpen, isPinned]);
  
  // Persist pinned state to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('sidebarPinned', isPinned.toString());
    }
  }, [isPinned]);
  
  // Handle focus detection for sidebar
  useEffect(() => {
    // Function to check if focus is within the sidebar
    const handleFocusCheck = () => {
      if (sidebarRef.current && sidebarRef.current.contains(document.activeElement)) {
        setSidebarVisible(true);
        
        // Clear any existing auto-hide timeout
        if (autoHideTimeoutRef.current) {
          clearTimeout(autoHideTimeoutRef.current);
          autoHideTimeoutRef.current = null;
        }
      }
    };
    
    // Set up listeners
    document.addEventListener('focusin', handleFocusCheck);
    return () => document.removeEventListener('focusin', handleFocusCheck);
  }, []);
  
  // Handle clicks outside the sidebar to close it (only if not pinned)
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        isSidebarVisible && 
        !isPinned && // Only close if not pinned
        sidebarRef.current && 
        !sidebarRef.current.contains(event.target as Node) &&
        !isHoveringTrigger
      ) {
        setSidebarVisible(false);
        announce({ 
          message: 'Sidebar closed', 
          politeness: 'polite' 
        });
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isSidebarVisible, isHoveringTrigger, isPinned]);
  
  // Handle escape key to close sidebar
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isSidebarVisible) {
        setSidebarVisible(false);
        announce({ 
          message: 'Sidebar closed', 
          politeness: 'polite' 
        });
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isSidebarVisible]);

  // Add focus outline styles when using keyboard navigation
  useEffect(() => {
    const handleFirstTab = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        document.body.classList.add('user-is-tabbing');
        window.removeEventListener('keydown', handleFirstTab);
      }
    };

    window.addEventListener('keydown', handleFirstTab);
    return () => {
      window.removeEventListener('keydown', handleFirstTab);
    };
  }, []);
  
  // Handle mouseenter on sidebar to clear auto-hide timeout
  useEffect(() => {
    const sidebarElement = sidebarRef.current;
    
    const handleMouseEnter = () => {
      if (autoHideTimeoutRef.current) {
        clearTimeout(autoHideTimeoutRef.current);
        autoHideTimeoutRef.current = null;
      }
    };
    
    sidebarElement?.addEventListener('mouseenter', handleMouseEnter);
    return () => {
      sidebarElement?.removeEventListener('mouseenter', handleMouseEnter);
    };
  }, []);
  
  // Handle pinning/unpinning sidebar
  const togglePinned = () => {
    const newPinnedState = !isPinned;
    setIsPinned(newPinnedState);
    
    // Ensure sidebar is visible when pinned
    if (newPinnedState) {
      setSidebarVisible(true);
      announce({ 
        message: 'Sidebar pinned open', 
        politeness: 'polite' 
      });
    } else {
      announce({ 
        message: 'Sidebar unpinned', 
        politeness: 'polite' 
      });
    }
  };
  
  // Navigation items
  const navItems = [
    ...(isAuthenticated
      ? [
          {
            path: '/chat',
            label: 'Chat',
            icon: (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
            ),
          },
          // Admin Dashboard link moved to user profile dropdown
        ]
      : [
          {
            path: '/login',
            label: 'Login',
            icon: (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" />
              </svg>
            ),
          },
          {
            path: '/register',
            label: 'Register',
            icon: (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
              </svg>
            ),
          },
        ]),
  ];

  const isActive = (path: string) => {
    if (path === '/chat' && router.pathname === '/chat') {
      return true;
    }
    if (path === '/admin' && router.pathname.startsWith('/admin')) {
      return true;
    }
    return router.pathname === path;
  };

  return (
    <>
      {/* Skip links for keyboard navigation - visually hidden until focused */}
      <div className="sr-only" aria-label="Accessibility navigation">
        <SkipLink target="main-content" label="Skip to main content" position="top-center" />
        <SkipLink target="nav-sidebar" label="Skip to navigation" position="top-center" />
      </div>
      
      <Head>
        <title>{title}</title>
        <meta name="description" content={description} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 flex">
        {/* Mobile backdrop overlay - only shown on mobile when sidebar is open */}
        {isSidebarVisible && (
          <div 
            className="md:hidden sidebar-backdrop"
            onClick={() => {
              setSidebarVisible(false);
              announce({ 
                message: 'Sidebar closed', 
                politeness: 'polite' 
              });
            }}
            aria-hidden="true"
          />
        )}
        
        {/* Sidebar trigger area - desktop only */}
        <div 
          className="fixed left-0 top-0 bottom-0 w-12 z-30 hidden md:block"
          onMouseEnter={() => {
            setIsHoveringTrigger(true);
            setSidebarVisible(true);
            announce({ 
              message: 'Sidebar opened', 
              politeness: 'polite' 
            });
          }}
          onMouseLeave={() => {
            setIsHoveringTrigger(false);
            // Only set auto-hide timeout if not pinned
            if (!isPinned) {
              autoHideTimeoutRef.current = setTimeout(() => {
                if (!sidebarRef.current?.contains(document.activeElement)) {
                  setSidebarVisible(false);
                  announce({ 
                    message: 'Sidebar closed', 
                    politeness: 'polite' 
                  });
                }
              }, 1000); // 1 second delay
            }
          }}
          aria-hidden="true"
        />
        
        {/* Accessible keyboard trigger button - visually hidden until focused */}
        <button
          className="fixed left-0 top-1/2 transform -translate-y-1/2 bg-gray-800 dark:bg-gray-900 text-white p-1 rounded-r-md focus:outline-none focus:ring-2 focus:ring-primary-500 sr-only focus:not-sr-only z-30"
          onClick={() => {
            setSidebarVisible(true);
            announce({ 
              message: 'Sidebar opened', 
              politeness: 'polite' 
            });
          }}
          aria-label="Open navigation sidebar"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
          </svg>
        </button>
        
        {/* Combined Navigation and Chat History Sidebar */}
          <div
          ref={sidebarRef}
          id="nav-sidebar"
          className={`fixed left-0 top-0 bottom-0 z-30 w-64 md:w-72 lg:w-80 bg-gray-800 dark:bg-gray-900 shadow-lg transition-transform duration-300 transform flex flex-col ${
            isSidebarVisible ? 'translate-x-0' : '-translate-x-full'
          } ${hideDefaultSidebar ? 'hidden' : ''}`}
          role={ariaLandmarks.navigation}
          aria-label="Main Navigation and Chat History"
          aria-expanded={isSidebarVisible}
        >
          {/* App Logo and Title */}
          <div className="p-4 flex items-center justify-between">
            <button
              onClick={togglePinned}
              className="absolute right-4 text-gray-300 hover:text-white"
              aria-label={isPinned ? "Unpin sidebar" : "Pin sidebar"}
            >
              {isPinned ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M6 4a2 2 0 012-2h4a2 2 0 012 2v14a2 2 0 01-2 2H8a2 2 0 01-2-2V4zm2 0v14h4V4H8z" clipRule="evenodd" />
                </svg>
              )}
            </button>
            <Link href="/" className="flex items-center py-2 w-full">
              <span className="text-white text-lg w-full text-center">Doogie Bot Chat</span>
            </Link>
          </div>
          
          <div className="flex-1 overflow-hidden flex flex-col">
            {/* Integrated Chat History and Navigation */}
              {/* Navigation Section - Display first above chat history */}
              <ul className="mb-6 w-full no-divider">
              <nav className="flex-grow overflow-y-auto p-4 w-full" aria-label="Main Menu">
                {navItems.map((item) => (
                  <li key={item.path} className="w-full mb-2">
                    <Link
                      href={item.path}
                      className={`flex items-center justify-center text-xs py-1.5 rounded-md w-full sidebar-button ${
                        isActive(item.path)
                          ? 'bg-primary-700 text-white'
                          : 'text-gray-300 hover:bg-gray-700'
                      }`}
                      aria-current={isActive(item.path) ? 'page' : undefined}
                    >
                      <span className="flex-shrink-0 h-4 w-4 mr-1" aria-hidden="true">
                        {typeof item.icon === 'object' && React.isValidElement(item.icon) 
                          ? React.cloneElement(item.icon, { className: 'h-4 w-4' } as React.HTMLAttributes<HTMLElement>)
                          : item.icon
                        }
                      </span>
                      <span>{item.label}</span>
                    </Link>
                  </li>
                ))}
              </nav>
              </ul>

              {/* Always display sidebar content (chat history) below navigation */}
              {sidebarContentToShow && (
                <div className="sidebar-chat-history w-full no-divider flex-1 overflow-y-auto">
                  {sidebarContentToShow}
                </div>
              )}
              
              {/* User profile section at bottom of sidebar */}
              {isAuthenticated && (
                <div className="mt-auto p-4 border-t border-gray-700 dark:border-gray-800">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center text-white">
                      {user?.email?.charAt(0).toUpperCase() || 'U'}
                    </div>
                    <div className="flex-1 truncate">
                      <p className="text-sm font-medium text-white truncate">{user?.email}</p>
                    </div>
                    <button
                      onClick={() => setShowUserMenu(!showUserMenu)}
                      className="text-gray-300 hover:text-white"
                      aria-label="User menu"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                  
                  {/* User menu dropdown */}
                  {showUserMenu && (
                    <div className="mt-2 py-1 bg-gray-700 rounded-md shadow-lg">
                      <Link
                        href="/profile"
                        className="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-600"
                        onClick={() => setShowUserMenu(false)}
                      >
                        <div className="flex items-center">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                          </svg>
                          Profile Settings
                        </div>
                      </Link>
                      
                      {/* Add Admin Dashboard link here if user is admin */}
                      {isAdmin && (
                        <Link
                          href="/admin"
                          className="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-600"
                          onClick={() => setShowUserMenu(false)}
                        >
                          <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                            </svg>
                            Admin Dashboard
                          </div>
                        </Link>
                      )}
                      <button
                        onClick={() => {
                          toggleTheme();
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-600"
                      >
                        <div className="flex items-center">
                          {theme === 'dark' ? (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                            </svg>
                          )}
                          {theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                        </div>
                      </button>
                      {shortcuts && (
                        <button
                          onClick={() => {
                            shortcuts.toggleShortcutDialog();
                            setShowUserMenu(false);
                          }}
                          className="block w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-600"
                        >
                          <div className="flex items-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                            </svg>
                            Keyboard Shortcuts
                          </div>
                        </button>
                      )}
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          logout();
                        }}
                        className="block w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-gray-600"
                      >
                        <div className="flex items-center">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                          </svg>
                          Sign Out
                        </div>
                      </button>
                    </div>
                  )}
                </div>
              )}
          </div>
        </div>
        
        {/* Main content */}
        <div className="flex-1 flex flex-col min-h-screen">
          {/* Main content area */}
          <main 
            className={`flex-1 flex flex-col overflow-auto min-h-0 transition-all duration-300 ${hideDefaultSidebar ? '' : isSidebarVisible ? 'md:ml-72 lg:ml-80' : 'md:ml-0'} w-auto ${hideDefaultSidebar ? '' : isSidebarVisible ? 'md:w-[calc(100%-18rem)] lg:w-[calc(100%-20rem)]' : 'w-full'}`}
            role={ariaLandmarks.main}
            id="main-content"
            ref={mainContentRef}
            tabIndex={-1}
            aria-label="Main content"
          >
            {/* App header for mobile - only shown on smaller screens */}
            {/* App header with three columns: menu button, centered title, and profile */}
            <div className="grid grid-cols-3 items-center mb-4 w-full p-6 pt-4">
              {/* Left section - mobile menu button */}
              <div className="flex items-center justify-start">
                {!hideDefaultSidebar && (
                  <button
                    onClick={() => {
                      setSidebarVisible(!isSidebarVisible);
                      announce({ 
                        message: isSidebarVisible ? 'Sidebar closed' : 'Sidebar opened', 
                        politeness: 'polite' 
                      });
                    }}
                    className="text-gray-700 dark:text-gray-300 mr-2 md:hidden"
                    aria-label={isSidebarVisible ? "Close navigation" : "Open navigation"}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                  </button>
                )}
              </div>
              
              {/* Center section - title */}
              <div className="flex justify-center items-center">
                {isEditingTitle ? (
                  <div className="max-w-xs">
                    <Input
                      ref={titleInputRef}
                      value={editedTitle}
                      onChange={(e) => setEditedTitle(e.target.value)}
                      onBlur={handleSaveTitle}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleSaveTitle();
                        } else if (e.key === 'Escape') {
                          setIsEditingTitle(false);
                        }
                      }}
                      className="font-bold text-lg"
                      aria-label="Edit chat title"
                    />
                  </div>
                ) : (
                  <h1 className="text-lg font-bold flex items-center group relative">
                    <img 
                      src="/images/logo.png" 
                      alt="Doogie Chat Bot Logo" 
                      className="h-14 w-auto mr-2" 
                      aria-hidden="true"
                    />
                    <span className="cursor-pointer group-hover:text-primary-600 dark:group-hover:text-primary-400">
                      {title}
                      {router.pathname.startsWith('/chat') && router.query.id && (
                        <button 
                          className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                          onClick={(e) => {
                            e.stopPropagation();
                            setIsEditingTitle(true);
                          }}
                          aria-label="Edit title"
                          title="Edit title"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                          </svg>
                        </button>
                      )}
                    </span>
                  </h1>
                )}
              </div>
              
              {/* Right section - profile dropdown */}
              <div className="flex items-center justify-end space-x-2">
                {/* Mobile-only user menu button */}
                {isAuthenticated && (
                  <button
                    className="md:hidden text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400"
                    onClick={() => setSidebarVisible(true)}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </button>
                )}
                {!isAuthenticated && (
                  <Link
                    href="/login"
                    className="text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400"
                  >
                    Login
                  </Link>
                )}
                {/* Export dropdown for chat pages */}
                {isAuthenticated && router.pathname.startsWith('/chat') && router.query.id && (
                  <ExportDropdown chat={{ id: String(router.query.id) }} />
                )}
              </div>
            </div>
            
            {children}
          </main>
          
          {/* Footer */}
          <footer 
            className="fixed bottom-0 left-0 right-0 p-2 text-center text-xs text-gray-500 dark:text-gray-400 z-10 pointer-events-none"
            role={ariaLandmarks.contentinfo}
          >
            <p>
              &copy; {new Date().getFullYear()} Doogie Chat Bot
            </p>
          </footer>
        </div>
      </div>
    </>
  );
};

export default Layout;