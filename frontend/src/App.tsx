import { useRef, useState } from 'react'
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
  type: 'human' | 'bot'
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

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      type: 'bot',
      content: "Hi there! I'm Emma, your friendly UIC chatbot assistant. I'm here to help you with any questions or concerns you may have as a student at the University of the Immaculate Conception. Let's work together to uphold our values of faith, excellence, and service. Have a great day!",
      actions: sampleQuestions.map((question) => ({
        command: 'input_question',
        label: question.question,
        payload: question
      }))
    },
  ])

  const inputRef = useRef<HTMLInputElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  const storeBotMessage = (message: string, status?: Message['status']) => {
    setMessages(msg => {
      const lastMessage = msg[msg.length - 1];

      if (lastMessage.type === 'bot' && lastMessage.status === 'loading') {
        return [...msg.slice(0, msg.length - 1), {
          ...lastMessage,
          content: message,
          status
        }]
      }

      return [...msg, {
        type: 'bot',
        content: message,
        status
      }]
    });
  }

  const onSubmit = async (input: string) => {
    if (!input) return null;

    setMessages(msg => [...msg,
      {
        type: 'human',
        content: input
      },
      {
        type: 'bot',
        content: '',
        status: 'loading'
      }
    ]);

    scrollToBottom();

    // sleep for 1 second
    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      const result = await chain.invoke({ input });
      storeBotMessage(result.answer);
      return result.answer;
    } catch (err) {
      console.error(err);
      storeBotMessage('Sorry, I encountered an error. Please try again.', 'error');
      return null;
    } finally {
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
    if (!inputRef.current) return;

    const value = inputRef.current.value;

    // reset input
    inputRef.current.value = '';
    onSubmit(value);
  }

  const onExecuteAction = (action: MessageAction) => {
    if (action.command === 'input_question') {
      onSubmit(action.payload.question);
    }
  }

  return (
    <main className="flex flex-col flex-nowrap h-screen">
      <header className="flex-none px-8 py-4 bg-primary-50 shadow flex items-center justify-center">
        <p className="font-bold">Emma</p>
      </header>

      <section ref={contentRef} className="max-w-5xl mx-auto flex-auto flex flex-col h-0 px-4 overflow-y-auto">
        <div className={cn("lg:-mx-8 px-4 py-4 space-y-4 w-full flex flex-col mt-auto justify-end")}>
          {messages.map((message, index) => (
            <div key={index} className={cn('w-full lg:w-3/4 flex flex-col', message.type === 'bot' ? 'items-start' : 'ml-auto items-end')}>
              <div className={cn("flex justify-start",
                message.status === 'loading' ? 'items-stretch' : 'items-start',
                message.type === 'bot' ? 'flex-row' : 'flex-row-reverse')}>
                <div
                  style={{ backgroundImage: `url(/default_avatar.png)` }}
                  className={cn("bg-cover bg-center bg-no-repeat w-12 h-12 border rounded-lg flex-shrink-0", message.type === 'bot' ? 'mr-2' : 'ml-2')}>
                </div>
                <div className={cn(
                  "px-4 py-2 rounded-lg",
                  {
                    'border border-danger-200 bg-danger-100': message.status === 'error',
                    'bg-primary-100': message.status !== 'error' && message.type === 'bot',
                    'bg-white border border-primary-200': message.type === 'human',
                  },
                )}>
                  {message.status == 'loading' ? (
                    <div className="flex space-x-2 py-3 items-center">
                      <div className="w-2 h-2 bg-primary-300 rounded-full animate-pulse" />
                      <div className="w-2 h-2 bg-primary-300 rounded-full animate-pulse" />
                      <div className="w-2 h-2 bg-primary-300 rounded-full animate-pulse" />
                    </div>
                  ) : (
                    <div className="prose prose-li:marker:text-primary-400 max-w-none" dangerouslySetInnerHTML={{__html: marked.parse(message.content)}}></div>
                  )}
                </div>
              </div>
              {(message.status != 'loading' && message.actions) && (
                <div className={cn("flex flex-col space-y-4 pt-4 pl-14", message.type === 'bot' ? 'items-start' : 'ml-auto items-end')}>
                  {message.actions.map((action, index) => (
                    <button key={index} onClick={() => onExecuteAction(action)}
                      className="transition-colors px-4 py-2 rounded-2xl text-sm text-left border border-primary-100 bg-primary-50 hover:bg-primary-100">
                        {action.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      <footer className="flex-none px-8 py-4 bg-primary-50 shadow">
        <div className="max-w-5xl mx-auto space-y-4">
          <div className="flex space-x-4">
            <input ref={inputRef} onKeyDown={(ev) => {
              if (ev.key === 'Enter') {
                ev.preventDefault();
                retrieveAndSubmit();
              }
            }} type="text" className="rounded-lg bg-white border px-4 py-4 flex-1" placeholder="What can I help you with?" />
            <button onClick={retrieveAndSubmit} className="transition-colors bg-primary-700 hover:bg-primary-800 text-white rounded-full px-8 py-4 font-bold">
              <svg className="w-6 h-6" viewBox="0 0 24 24">
                <path fill="currentColor" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>

          {/* Add disclaimer */}
          <p className="text-sm text-center text-gray-500">This chatbot is for demonstration purposes only and may not be totally accurate. It is not a replacement for official UIC personnel.</p>
        </div>
      </footer>
    </main>
  )
}

export default App
