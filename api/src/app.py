import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .diagnose import diagnose

REFERENCE_TIMES_PATH = Path(__file__).parent.parent / "reference_times.json"

reference_times: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global reference_times
    with open(REFERENCE_TIMES_PATH) as f:
        reference_times = json.load(f)
    yield


app = FastAPI(title="VaultTech Diagnostics API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    msg = exc.errors()[0]["msg"] if exc.errors() else "invalid request body"
    return JSONResponse(status_code=400, content={"error": msg})


class PieceRequest(BaseModel):
    piece_id: str
    die_matrix: int
    lifetime_2nd_strike_s: Optional[float] = None
    lifetime_3rd_strike_s: Optional[float] = None
    lifetime_4th_strike_s: Optional[float] = None
    lifetime_auxiliary_press_s: Optional[float] = None
    lifetime_bath_s: Optional[float] = None


@app.post("/diagnose")
def diagnose_endpoint(piece: PieceRequest):
    if str(piece.die_matrix) not in reference_times:
        return JSONResponse(
            status_code=400,
            content={"error": f"unknown die_matrix {piece.die_matrix}"},
        )
    result = diagnose(piece.model_dump(), reference_times)
    return result
