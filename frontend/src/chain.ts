// Using LangServe does not work as the chain input was not inferred correctly.
// Instead, we use a custom chain implementation that uses fetch.
const chain = (() => {
    const url = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

    async function invoke(input: { input: string }) {
        const res = await fetch(new URL("/invoke", url), {
            method: "POST",
            body: JSON.stringify({
                config: {},
                kwargs: {},
                input
            }),
            headers: {
                "Content-Type": "application/json",
            },
        });

        return await res.json();
    }

    return { invoke };
})()

export default chain;
