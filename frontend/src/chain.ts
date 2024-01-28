import { RemoteRunnable } from "langchain/runnables/remote";

const chain = new RemoteRunnable<{ input: string }, { answer: string }, never>({
    url: import.meta.env.VITE_API_URL ?? "http://localhost:8080",
});

export default chain;
