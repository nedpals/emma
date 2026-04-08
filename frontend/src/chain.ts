import { fetchEventSource } from '@microsoft/fetch-event-source';

type ChainEvent =
    | { type: "tool_start"; tool: string; arguments: Record<string, unknown> }
    | { type: "tool_end"; tool: string; success: boolean }
    | { type: "answer_chunk"; chunk: string }
    | { type: "answer_done" }
    | { type: "error"; message: string };

type InvokeCallbacks = {
    onEvent?: (event: ChainEvent) => void;
};

const chain = (() => {
    const url = import.meta.env.VITE_API_URL ?? window.location.origin;

    async function invoke(
        input: { input: string; chat_history: { role: string; content: string }[] },
        callbacks?: InvokeCallbacks,
    ): Promise<{ answer: string }> {
        let answer = "";
        let error: Error | null = null;

        await fetchEventSource(new URL("/invoke", url).toString(), {
            method: "POST",
            body: JSON.stringify({
                config: {},
                kwargs: {},
                input,
            }),
            headers: {
                "Content-Type": "application/json",
            },
            onmessage(ev) {
                if (!ev.data) return;
                let event: ChainEvent;
                try {
                    event = JSON.parse(ev.data);
                } catch {
                    return;
                }
                callbacks?.onEvent?.(event);

                if (event.type === "answer_chunk") {
                    answer += event.chunk;
                } else if (event.type === "error") {
                    error = new Error(event.message);
                }
            },
            onerror(err) {
                throw err;
            },
        });

        if (error) throw error;
        return { answer };
    }

    return { invoke };
})();

export type { ChainEvent, InvokeCallbacks };
export default chain;
