import os
from server import run_server

if __name__ == "__main__":
    run_server(
        os.environ.get("HOST", "localhost"),
        int(os.environ.get("PORT", "8000"))
    )
