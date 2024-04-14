import asyncio
import os

import psutil
import torch.cuda
import uvicorn
from fastapi import FastAPI, APIRouter
import server_basic_routers

process = psutil.Process(os.getpid())

app = FastAPI(openapi_url="/api/v1/openapi.json")

router = APIRouter()
router.include_router(server_basic_routers.router)
app.include_router(router)


@app.get("/health", tags=["health"])
def health():
    """
        Проверка доступности сервера.
    """
    return {
        "status": "OK",
        "info": {
            "mem": f"{process.memory_info().rss / (1024 ** 2):.3f} MiB",
            "cpu_usage": process.cpu_percent(),
            "threads": len(process.threads()),
            "cuda_is_available": torch.cuda.is_available(),
        }
    }


# http://localhost:9027/docs

async def run_server():
    config = uvicorn.Config(f"{__name__}:app",
                            host='0.0.0.0', port=9027,
                            log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == '__main__':
    asyncio.run(run_server())
