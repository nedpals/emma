import { useRef, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import chain from './client';

interface Message {
  type: 'human' | 'bot';
  content: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    // {
    //   type: 'human',
    //   content: 'Hello, I am a human'
    // },
    // {
    //   type: 'bot',
    //   content: 'Hello, I am a bot'
    // }
  ])

  const inputRef = useRef<HTMLInputElement>(null);

  const onSubmit = async (input: string) => {
    if (!input) return null;

    setMessages(msg => [...msg, {
      type: 'human',
      content: input
    }]);

    const result = await chain.invoke({ input });
    return result.answer;
  }

  const retrieveAndSubmit = () => {
    if (!inputRef.current) return;

    const value = inputRef.current.value;

    // reset input
    inputRef.current.value = '';

    onSubmit(value)
      .then((res) => {
        if (!res) return;

        setMessages(msg => [...msg, {
          type: 'bot',
          content: res
        }]);
      });
  }

  return (
    <main className="flex flex-col flex-nowrap h-screen">
      <header className="px-8 py-4 bg-pink-50 shadow flex items-center justify-center">
        <p>UIC Handbook Assistant</p>
      </header>

      <section className="max-w-5xl mx-auto px-8 py-4 space-y-4 w-full flex-1 flex flex-col justify-end overflow-y-auto">
        {messages.map((message, index) => (
          <div key={index} className={'flex flex-col justify-start ' + (message.type === 'bot' ? 'items-start' : 'items-end')}>
            <div className="flex flex-row items-center justify-start space-x-4">
              <img src={message.type === 'human' ? reactLogo : viteLogo} className="w-8 h-8" />
              <p className="px-4 py-2 rounded-lg bg-pink-100">{message.content}</p>
            </div>
          </div>
        ))}
      </section>

      <footer className="px-8 py-4 bg-pink-50">
        <div className="max-w-5xl mx-auto flex space-x-4">
          <input ref={inputRef} onKeyDown={(ev) => {
            if (ev.key === 'Enter') {
              ev.preventDefault();
              retrieveAndSubmit();
            }
          }} type="text" className="rounded-lg bg-white border px-4 py-4 flex-1" placeholder="Input your query" />
          <button onClick={retrieveAndSubmit} className="bg-pink-700 hover:bg-pink-800 text-white rounded-lg px-8 py-4 font-bold">Send</button>
        </div>
      </footer>
    </main>
  )
}

export default App
