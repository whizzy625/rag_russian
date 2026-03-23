import uvicorn

from config import WEB_HOST, WEB_PORT, WEB_RELOAD


if __name__ == "__main__":
    uvicorn.run("api:app", host=WEB_HOST, port=WEB_PORT, reload=WEB_RELOAD)
