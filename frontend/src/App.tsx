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

import * as React from "react";

const Logo = ({ showIcon = false, ...props }: React.SVGProps<SVGSVGElement> & { showIcon?: boolean }) => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 289 114" fill="none" {...props}>
    {/* Words */}
    <g clipPath="url(#a)" className={cn(
      "transition-all duration-300",
      // move words to the center if showIcon is false
      showIcon ? "translate-x-0" : "translate-x-8"
    )}>
      <path
        fill="#58162B"
        d="M24 80.302q-10.32 0-15.32-4.68-4.96-4.68-4.96-13.6v-4.76q0-8.84 4.64-13.56 4.68-4.72 13.6-4.72 6 0 10 2.08t6 5.92q2.04 3.84 2.04 9.2v1.36q0 1.48-.16 3-.12 1.521-.36 2.88h-9.92q.12-2.28.16-4.28t.04-3.6q0-2.68-.84-4.48-.84-1.84-2.56-2.8t-4.4-.96q-3.96 0-5.84 2.16-1.84 2.16-1.84 6.2v3.6l.04 1.2v3q0 1.72.56 3.2t1.88 2.6q1.32 1.08 3.56 1.72 2.241.6 5.64.6 3.68 0 7-.76a38.3 38.3 0 0 0 6.4-2.12l-.92 8.48q-2.679 1.44-6.36 2.28-3.64.84-8.08.84M9.52 63.422v-7.08h27.76v7.08zm81.835 15.92v-23.72q0-2.24-.56-3.88-.56-1.68-1.88-2.56-1.32-.92-3.6-.92-2 0-3.52.8-1.48.76-2.44 2.08-.96 1.28-1.4 2.96l-1.16-5.64h1.08q.64-2.56 2.04-4.64 1.44-2.12 3.92-3.36 2.52-1.28 6.36-1.28 4.24 0 6.92 1.68 2.68 1.64 3.92 4.96 1.28 3.28 1.28 8.16v25.36zm-48.32 0v-39.24h10.96l-.4 10.16.44.4v28.68zm24.16 0v-23.72q0-2.24-.56-3.88-.56-1.68-1.88-2.56-1.32-.92-3.56-.92-2.04 0-3.52.8-1.48.76-2.44 2.08-.96 1.28-1.44 2.96l-1.72-5.64h2q.6-2.64 2-4.72 1.401-2.08 3.84-3.32t6.12-1.24q5.48 0 8.24 2.84 2.8 2.8 3.52 8.16.12.76.24 1.88.12 1.08.12 2v25.28zm87.738 0v-23.72q0-2.24-.56-3.88-.56-1.68-1.88-2.56-1.32-.92-3.6-.92-2 0-3.52.8-1.48.76-2.44 2.08-.96 1.28-1.4 2.96l-1.16-5.64h1.08q.64-2.56 2.04-4.64 1.44-2.12 3.92-3.36 2.52-1.28 6.36-1.28 4.24 0 6.92 1.68 2.68 1.64 3.92 4.96 1.28 3.28 1.28 8.16v25.36zm-48.32 0v-39.24h10.96l-.4 10.16.44.4v28.68zm24.16 0v-23.72q0-2.24-.56-3.88-.56-1.68-1.88-2.56-1.32-.92-3.56-.92-2.04 0-3.52.8-1.48.76-2.44 2.08-.96 1.28-1.44 2.96l-1.72-5.64h2q.6-2.64 2-4.72 1.401-2.08 3.84-3.32t6.12-1.24q5.48 0 8.24 2.84 2.8 2.8 3.52 8.16.12.76.24 1.88.12 1.08.12 2v25.28zm62.099 0 .36-9.68-.28-.8v-12.28l-.04-1.68q0-3.64-2.04-5.36t-6.72-1.72q-4.04 0-7.6 1.04a52 52 0 0 0-6.56 2.4l.92-8.64q1.8-.879 4.08-1.68 2.32-.84 5.12-1.32 2.8-.52 6.04-.52 4.96 0 8.32 1.16t5.36 3.32q2.04 2.12 2.92 5.12.92 2.961.92 6.6v24.04zm-12.72.92q-5.88 0-8.92-2.92-3.04-2.96-3.04-8.36v-1.12q0-5.76 3.52-8.48 3.56-2.76 11.24-3.8l10.92-1.48.64 6.88-9.92 1.4q-3.2.44-4.52 1.52-1.28 1.08-1.28 3.2v.36q0 2.04 1.28 3.2 1.32 1.16 4.12 1.16 2.48 0 4.28-.76 1.8-.759 2.92-2.04a8.3 8.3 0 0 0 1.68-2.88l1.56 5.04h-1.88a14 14 0 0 1-2 4.6q-1.401 2.04-3.96 3.28-2.52 1.2-6.64 1.2"
      ></path>
    </g>
    {/* Icon */}
    <g className={cn(
      "transition-all duration-300",
      showIcon ? "translate-x-0 opacity-100" : "translate-x-4 opacity-0"
    )}>
      <path
        fill="#B42D4F"
        d="M269.255 25.578a.497.497 0 0 1 .9-.229l3.316 4.761a.5.5 0 0 0 .356.21l5.799.599a.49.49 0 0 1 .23.893l-4.794 3.294a.5.5 0 0 0-.208.326l-.003.027-.604 5.758c-.047.454-.638.604-.899.229l-3.317-4.761a.5.5 0 0 0-.356-.21l-5.798-.599a.492.492 0 0 1-.231-.893l4.795-3.293a.5.5 0 0 0 .211-.354z"
      ></path>
      <path
        fill="#B42D4F"
        fillRule="evenodd"
        d="M254.614 17.304c-22.237 0-33.75 14.029-33.75 31.333l.234 15.23c.313 20.39 19.06 35.297 38.992 30.386 3.452-.85 6.405-1.548 8.976-2.04l-3.753-5.258c1.665-1.513 3.868-4.173 5.246-6.864-3.024 2.535-7.225 4.122-7.225 4.122a26 26 0 0 1-8.72 1.499c-14.261 0-25.822-11.48-25.822-25.641v-6.497a4.84 4.84 0 0 1 3.596-4.693q.99-.26 1.952-.502l-.014.001.163-.038q.947-.236 1.871-.452a2 2 0 0 0 1.276-1.464l1.55-7.31.355 5.75c.071 1.148 1.1 2 2.25 1.862l.535-.065c12.122-2.115 21.597-1.255 34.529 2.21a4.845 4.845 0 0 1 3.581 4.691v6.507c0 5.524-1.76 10.64-4.752 14.826l-4.35 16.921c6.379-.982 10.2-.334 13.818 2.97 1.142 1.044 3.212.346 3.212-1.195V48.637c0-8.44-2.836-16.1-8.274-21.733l-.033.317c-.17 1.627-2.288 2.166-3.225.82l-.873-1.254-1.527-.157c-1.639-.17-2.182-2.272-.827-3.203l1.199-.823c-5.282-3.347-12.047-5.3-20.19-5.3m16.865 7.136c-1.112-1.596-3.623-.957-3.825.973l-.551 5.251-4.373 3.004c-1.607 1.104-.964 3.598.98 3.798l5.289.547 3.025 4.342c1.111 1.596 3.623.958 3.825-.972l.55-5.252 4.373-3.004c1.607-1.104.964-3.597-.979-3.798l-5.289-.547z"
        clipRule="evenodd"
      ></path>
      <path
        fill="#B42D4F"
        d="M274.521 24.742a.164.164 0 0 0 .077.297l2.188.227q.064.007.108.055l.011.014 1.251 1.797a.165.165 0 0 0 .3-.077l.228-2.172a.16.16 0 0 1 .07-.118l1.809-1.242a.164.164 0 0 0-.077-.298l-2.188-.226a.16.16 0 0 1-.118-.07l-1.252-1.797a.165.165 0 0 0-.3.077l-.227 2.172a.16.16 0 0 1-.071.118z"
      ></path>
      <path
        fill="#B42D4F"
        d="M241.834 62c2.232 0 4.04-1.68 4.04-3.75s-1.808-3.751-4.04-3.751c-2.231 0-4.039 1.68-4.039 3.75 0 2.072 1.808 3.751 4.039 3.751"
      ></path>
      <path
        fill="#B42D4F"
        d="m262.292 68.194-14.62 2.435a.798.798 0 0 0-.533 1.238l2.302 3.358c.171.249.473.381.774.342 8.079-1.062 10.788-1.945 12.915-6.298.275-.563-.216-1.178-.838-1.075"
      ></path>
      <path
        fill="#B42D4F"
        d="M260.318 59.306c2.278-1.214 3.807-1.676 5.125-1.656 1.298.02 2.573.509 4.347 1.615.565.352 1.31.183 1.665-.378.354-.56.184-1.3-.381-1.653-1.916-1.195-3.644-1.953-5.595-1.982-1.932-.029-3.9.66-6.303 1.941a1.195 1.195 0 0 0-.493 1.624 1.21 1.21 0 0 0 1.635.489"
      ></path>
    </g>
  </svg>
);

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
            <img src="/star.png" alt="Emma Star" className="w-6 h-auto p-0.5" />
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
    <div className="absolute inset-0 flex items-center justify-center p-4 bg-gradient-to-b from-primary-25/95 via-primary-50/95 to-primary-100/95 animate-fadeIn pt-16">
      <div className="max-w-2xl w-full">
        <div className="text-center mb-8">
          <img src="/emma_icon.png" alt="Emma Logo" className="w-32 h-auto mx-auto mb-8" />
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
        "flex-none flex items-center px-6 py-1.5 transition-colors duration-200 justify-center",
        "fixed w-full inset-x-0 z-50",
        isHeaderTransparent ? "bg-transparent" : "bg-primary-50/60 border-b border-primary-100/60 backdrop-blur-md"
      )}>
        <button onClick={() => {
          setHasStarted(false);
          setMessages([]);
          setIsInputEmpty(true);
          setIsSubmitting(false);
          inputRef.current!.value = '';
          adjustTextareaHeight();
        }}>
          <Logo className="w-32 h-auto" showIcon={hasStarted} />
        </button>
      </header>

      {/* Chat content */}
      <section 
        ref={contentRef} 
        className="flex-auto flex flex-col h-0 overflow-y-auto px-4 md:px-6 scrollbar-thin scrollbar-thumb-primary-200/50 scrollbar-track-transparent relative pt-12"
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
