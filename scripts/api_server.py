import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import os
from dotenv import load_dotenv
import sys


load_dotenv()


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_fixer_system.map_fixer import generate_level, bfs_reachability


DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI(
    title="Mamba Mario Generator API",
    description="REST API to generate Mario levels using a Conditional Mamba model."
)



class AttributeParams(BaseModel):
    enemies: int = 2
    gaps: int = 1
    pipes: int = 1

class GenerationParams(BaseModel):
    num_columns: int = 120
    temperature: float = 0.8
    top_k: int = 30
    top_p: float = 1.5  
    cfg_scale: float = 1.3
    seed: Optional[int] = None

class GenerationRequest(BaseModel):
    attributes: AttributeParams
    params: GenerationParams
    api_key: Optional[str] = DEFAULT_GROQ_KEY

class GenerationResponse(BaseModel):
    status: str
    width: int
    height: int
    level_string: str
    solvable: bool
    fixed: bool = False
    fix_rounds: int = 0
    generation_time_ms: float



@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    
    start_time = time.time()
    
    try:

        attr_list = [
            float(request.attributes.enemies),
            float(request.attributes.gaps),
            float(request.attributes.pipes)
        ]
        


        grid = generate_level(
            attributes=attr_list,
            patches=request.params.num_columns / 16.0,
            seed=request.params.seed,
            temperature=request.params.temperature,
            top_k=request.params.top_k,
            top_p=request.params.top_p,
            cfg_scale=request.params.cfg_scale
        )
        

        final_grid = grid
        is_solvable = False
        fixed_status = False
        rounds_taken = 0
        
        if request.api_key and request.api_key.strip():

            print("Running Map Fixing Pipeline...")
            from map_fixer_system.map_fixer import run_fix_pipeline
            
            last_state = None
            for update in run_fix_pipeline(grid, api_key=request.api_key):
                if isinstance(update, dict) and "grid" in update:
                    print("Fixing Round:", update["round"])
                    last_state = update
            
            if last_state:
                final_grid = last_state["grid"]
                is_solvable = last_state["solvable"]
                rounds_taken = last_state["round"]
                fixed_status = rounds_taken > 0
        else:

            _, _, is_solvable, _ = bfs_reachability(grid)
        

        level_string = "\n".join(["".join(row) for row in final_grid])
        
        end_time = time.time()
        
        return GenerationResponse(
            status="success",
            width=len(final_grid[0]),
            height=len(final_grid),
            level_string=level_string,
            solvable=is_solvable,
            fixed=fixed_status,
            fix_rounds=rounds_taken,
            generation_time_ms=(end_time - start_time) * 1000
        )
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    
    return {"status": "ready", "model": "Conditional-Mamba-Mario-v1"}

if __name__ == "__main__":
    print("Starting Mamba Mario API Server...")

    uvicorn.run(app, host="0.0.0.0", port=8000)
