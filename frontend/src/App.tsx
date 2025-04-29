import { useRef, useState, useEffect } from 'react'
import chain from './chain';

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { marked } from 'marked';

function cn(...args: ClassValue[]) {
  return twMerge(clsx(args));
}

type MessageAction = {
  command: 'input_question',
  label: string,
  payload: { question: string }
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  status?: 'loading' | 'error'
  actions?: MessageAction[]
}

const sampleQuestions = [
  {
    "question": "What are the maximum number of unexcused absences before getting dropped from a course?"
  },
  {
    "question": "How many tardies equal one absence?"
  },
  {
    "question": "What is the procedure for getting permission to miss a class for extra-curricular activities?"
  }
]

interface MessageBubbleProps {
  role: Message['role']
  content: string
  status?: Message['status']
  actions?: MessageAction[]
  onActionClick?: (action: MessageAction) => void
}

function AnimatedTextRenderer({ 
  text, 
  className,
  status 
}: { 
  text: string, 
  className?: string,
  status?: Message['status']
}) {
  const [displayedText, setDisplayedText] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const textRef = useRef(text);
  
  useEffect(() => {
    if (status === 'error') {
      // For error messages, show them immediately
      setDisplayedText(text);
      setIsComplete(true);
      return;
    }
    
    textRef.current = text;
    setIsComplete(false);
    
    // Split and process Markdown
    if (text) {
      setDisplayedText(""); // Reset before animation starts
      
      // Use a small delay before starting animation
      const initialDelay = setTimeout(() => {
        let position = 0;
        const increment = 1; // Characters per tick
        const speedBase = 5; // Base speed in milliseconds
        
        const timer = setInterval(() => {
          if (position < textRef.current.length) {
            setDisplayedText(textRef.current.substring(0, position + increment));
            position += increment;
          } else {
            clearInterval(timer);
            setIsComplete(true);
          }
        }, speedBase);
        
        return () => clearInterval(timer);
      }, 100);
      
      return () => clearTimeout(initialDelay);
    }
  }, [text, status]);
  
  if (!text) return null;
  
  return (
    <div className={className}>
      {isComplete ? (
        // Once animation is complete, use marked for proper rendering
        <div
          className="prose prose-sm max-w-none prose-p:leading-relaxed prose-p:my-1 prose-ul:my-2 prose-li:my-0.5"
          dangerouslySetInnerHTML={{ __html: marked.parse(text, { breaks: true }) }}
        />
      ) : (
        // During animation, render plaintext with word-by-word animation
        <div className="prose prose-sm max-w-none prose-p:leading-relaxed">
          {displayedText.split(' ').map((word, i) => (
            <span key={i} className="animate-fadeIn" style={{ 
              animationDelay: `${i * 30}ms`,
              animationDuration: '200ms',
              display: 'inline'
            }}>
              {word}{' '}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function MessageBubble({ role, content, status, actions, onActionClick }: MessageBubbleProps) {
  return (
    <div className="flex flex-col gap-1.5 w-full mb-6">
      {/* Speaker label */}
      {role === 'user' && 
        <div className="text-xs font-medium text-primary-800/70 mb-0.5">
          You
        </div>}
      
      {/* Message content */}
      <div className="flex items-start gap-3">
        {role === 'assistant' && (
          <div className="w-11 h-11 rounded-xl bg-primary-600 flex items-center justify-center flex-shrink-0 mt-0.5">
            <span className="text-white font-semibold text-sm">E</span>
          </div>
        )}
        
        <div className={cn(
          role === 'assistant' && status === 'loading' && "min-w-[60px] min-h-[40px]",
          {
            'text-primary-900 w-full': role === 'assistant' && status !== 'error',
            'bg-danger-50/50 px-4 py-3 rounded-xl border border-danger-100/50 w-full': status === 'error',
            'bg-gradient-to-br from-primary-100/95 to-primary-200/80 px-4 py-3 rounded-xl max-w-fit border border-primary-100': role === 'user',
            'px-2 py-3': role === 'assistant' && status === undefined,
            'w-fit': role === 'user' && content.length < 300, // Fit content for short messages
            'w-full': role === 'user' && content.length >= 300 // Full width for longer messages
          } 
        )}>
          {status === 'loading' ? (
            <div className="flex items-center space-x-2 h-full px-2 py-5">
              <div className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 rounded-full bg-primary-400 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          ) : (
            role === 'assistant' && status !== 'error' ? (
              <AnimatedTextRenderer 
                text={content} 
                status={status}
                className={cn(
                  "text-primary-900",
                )}
              />
            ) : (
              <div className={cn(
                "prose prose-sm max-w-none", 
                "prose-p:leading-relaxed prose-p:my-1 prose-ul:my-2 prose-li:my-0.5",
                role === 'user' ? 'text-primary-900' : 'text-primary-900',
                status === 'error' ? 'prose-headings:text-danger-700 prose-a:text-danger-700' : ''
              )}
              dangerouslySetInnerHTML={{ __html: marked.parse(content, { breaks: true }) }}
              />
            )
          )}
        </div>
      </div>

      {/* Follow-up actions */}
      {role === 'assistant' && status !== 'loading' && actions && (
        <div className="mt-2 ml-10 flex flex-wrap gap-2">
          {actions.map((action, idx) => (
            <button
              key={idx}
              onClick={() => onActionClick?.(action)}
              className="transition-colors px-3 py-1.5 rounded-lg text-xs 
                       bg-white hover:bg-primary-50 border border-primary-200/50
                       text-primary-700 hover:text-primary-800 hover:border-primary-300/50"
            >
              {action.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function WelcomeOverlay({ onActionClick }: { onActionClick: (action: MessageAction) => void }) {
  const getTimeBasedGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Maayong buntag";
    if (hour < 18) return "Maayong hapon";
    return "Maayong gabi-i";
  };

  return (
    <div className="absolute inset-0 flex items-center justify-center p-4 bg-gradient-to-b from-primary-25/95 via-primary-50/95 to-primary-100/95 animate-fadeIn">
      <div className="max-w-2xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-primary-900 mb-4">{getTimeBasedGreeting()}!<br className="md:hidden" /> What can I do for you?</h1>
          <div className="prose prose-md mx-auto text-primary-900/70">
            <p>Hi! I'm Emma, your AI companion for everything UIC. Whether you need guidance on academic policies, 
              campus life, or student services, I'm here to assist you. Together, let's uphold the cherished values of 
              faith, excellence, and service.</p>
          </div>
        </div>

        <div className="bg-white/40 rounded-2xl border border-primary-200/50 overflow-hidden">
          <div className="px-6 py-5">
            <p className="text-sm font-medium text-primary-900/70 mb-4">Questions that can get you started:</p>
            <div className="grid grid-cols-1 gap-2">
              {sampleQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => onActionClick({
                    command: 'input_question',
                    label: question.question,
                    payload: question
                  })}
                  className="transition-all px-4 py-3 rounded-xl text-sm text-left
                           bg-white/60 hover:bg-white/80 border border-primary-100/50
                           hover:border-primary-200/70 hover:shadow-sm
                           text-primary-900/80 font-medium flex items-center group"
                >
                  <span className="mr-3 text-primary-600">
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24">
                      <path fill="currentColor" d="M20 6h-1v8c0 .55-.45 1-1 1H6v1c0 1.1.9 2 2 2h10l4 4V8c0-1.1-.9-2-2-2zm-3 5V4c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v13l4-4h9c1.1 0 2-.9 2-2z"/>
                    </svg>
                  </span>
                  <span className="pr-1">{question.question}</span>
                  <svg xmlns="http://www.w3.org/2000/svg" 
                      className="w-4 h-4 ml-auto text-gray-400 group-hover:text-primary-600 group-hover:translate-x-1 transition-all" 
                      viewBox="0 0 24 24">
                    <path fill="currentColor" d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z"/>
                  </svg>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [hasStarted, setHasStarted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isInputEmpty, setIsInputEmpty] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isHeaderTransparent, setIsHeaderTransparent] = useState(true);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      if (!contentRef.current) return;
      setIsHeaderTransparent(contentRef.current.scrollTop < 10);
    };

    const contentElement = contentRef.current;
    if (contentElement) {
      contentElement.addEventListener('scroll', handleScroll);
      return () => contentElement.removeEventListener('scroll', handleScroll);
    }
  }, []);

  const storeBotMessage = (message: string, status?: Message['status']) => {
    setMessages(msg => {
      const lastMessage = msg[msg.length - 1];

      if (lastMessage.role === 'assistant' && lastMessage.status === 'loading') {
        return [...msg.slice(0, msg.length - 1), {
          ...lastMessage,
          content: message,
          status
        }]
      }

      return [...msg, {
        role: 'assistant',
        content: message,
        status
      }]
    });
  }

  const onSubmit = async (input: string) => {
    if (!input) return null;
    
    setIsSubmitting(true);
    setHasStarted(true);
    
    setMessages(msg => [...msg,
      {
        role: 'user',
        content: input
      },
      {
        role: 'assistant',
        content: '',
        status: 'loading'
      }
    ]);

    scrollToBottom();

    try {
      const history = messages.filter(msg => msg.status == null);
      const result = await chain.invoke({ input, chat_history: history });
      storeBotMessage(result.answer);
      return result.answer;
    } catch (err) {
      console.error(err);
      storeBotMessage('Sorry, I encountered an error. Please try again.', 'error');
      return null;
    } finally {
      setIsSubmitting(false);
      scrollToBottom();
    }
  }

  const scrollToBottom = () => {
    setTimeout(() => {
      if (!contentRef.current) {
        return;
      }

      if (contentRef.current.scrollTo) {
        contentRef.current.scrollTo({
          top: contentRef.current.scrollHeight,
          behavior: 'smooth'
        });
      } else {
        contentRef.current.scrollTop = contentRef.current.scrollHeight;
      }
    }, 50);
  }

  const retrieveAndSubmit = () => {
    if (!inputRef.current || isSubmitting) return;

    const value = inputRef.current.value.trim();
    if (!value) return;

    // reset input
    inputRef.current.value = '';
    setIsInputEmpty(true);
    adjustTextareaHeight();
    onSubmit(value);
  }

  // Adjust textarea height based on content
  const adjustTextareaHeight = () => {
    const textarea = inputRef.current;
    if (!textarea) return;
    
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
  };

  // Handle input changes
  const handleInputChange = () => {
    if (!inputRef.current) return;
    setIsInputEmpty(inputRef.current.value.trim() === '');
    adjustTextareaHeight();
  };

  const onExecuteAction = (action: MessageAction) => {
    if (action.command === 'input_question') {
      setHasStarted(true);
      onSubmit(action.payload.question);
    }
  }
  
  // Initialize textarea height
  useEffect(() => {
    adjustTextareaHeight();
  }, []);

  return (
    <main className="flex flex-col h-screen bg-gradient-to-br from-primary-25 via-primary-50/80 to-primary-100/50">
      {/* Header */}
      <header className={cn(
        "flex-none flex items-center px-6 py-3.5 transition-colors duration-200",
        "border-b border-primary-100/60 relative z-10",
        isHeaderTransparent ? "bg-transparent" : "bg-primary-50/60 backdrop-blur-md"
      )}>
        <button onClick={() => {
          setHasStarted(false);
          setMessages([]);
          setIsInputEmpty(true);
          setIsSubmitting(false);
          inputRef.current!.value = '';
          adjustTextareaHeight();
        }} className="flex justify-center space-x-4 max-w-3xl w-full mx-auto">
          <span className="text-lg font-medium text-primary-900">Emma</span>
        </button>
      </header>

      {/* Chat content */}
      <section 
        ref={contentRef} 
        className="flex-auto flex flex-col h-0 overflow-y-auto px-4 md:px-6 scrollbar-thin scrollbar-thumb-primary-200/50 scrollbar-track-transparent relative"
      >
        {!hasStarted && <WelcomeOverlay onActionClick={onExecuteAction} />}
        <div className="max-w-3xl w-full mx-auto flex flex-col py-6">
          {messages.map((message, index) => (
            <div key={index} className="animate-fadeIn">
              <MessageBubble
                role={message.role}
                content={message.content}
                status={message.status}
                actions={message.actions}
                onActionClick={onExecuteAction}
              />
            </div>
          ))}
        </div>
      </section>

      {/* Footer with input area */}
      <footer className="flex-none border-t border-primary-100/60 bg-primary-50/60 backdrop-blur-md px-4 py-4 relative z-10">
        <div className="max-w-3xl mx-auto">
          <div className="relative bg-white/80 border border-primary-200/60 rounded-2xl shadow-sm overflow-hidden focus-within:ring-2 focus-within:ring-primary-500/50 focus-within:border-primary-500/50">
            <textarea
              ref={inputRef}
              onChange={handleInputChange}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  retrieveAndSubmit();
                }
              }}
              placeholder="Ask me anything about UIC..."
              disabled={isSubmitting}
              rows={1}
              className="block w-full px-4 py-3 pr-24 resize-none focus:outline-none bg-transparent text-gray-700 placeholder-gray-500 text-sm"
            ></textarea>
            <button
              onClick={retrieveAndSubmit}
              disabled={isInputEmpty || isSubmitting}
              className={cn(
                "absolute right-2 top-1/2 -translate-y-1/2 py-2 transition-all flex items-center space-x-2",
                "bg-primary-600 hover:bg-primary-700 disabled:bg-gray-100",
                "text-white disabled:text-gray-400 text-sm font-medium",
                "group overflow-hidden rounded-xl",
                isInputEmpty || isSubmitting ? "px-3 w-10" : "px-4 w-[5.5rem]"
              )}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24">
                <path fill="currentColor" d="M3 20V4l19 8zm2-3l11.85-5L5 7v3.5l6 1.5l-6 1.5zm0 0V7z"></path>
              </svg>
              <span className={cn(
                "opacity-0 flex-shrink-0 group-enabled:slide-in-right",
                !isInputEmpty && !isSubmitting && "opacity-100"
              )}>Send</span>
            </button>
          </div>
          <p className="text-xs text-center text-primary-800/60 mt-3">
            Emma is an AI assistant trained to help with UIC-related inquiries. For official matters, please consult UIC personnel.
          </p>
        </div>
      </footer>
    </main>
  )
}

export default App
