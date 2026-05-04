import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import os
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Add project root to path so we can import internal modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_fixer_system.map_fixer import generate_level, bfs_reachability

# Default GROQ key from environment
DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

app = FastAPI(
    title="Mamba Mario Generator API",
    description="REST API to generate Mario levels using a Conditional Mamba model."
)

# --- Data Models ---

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

# --- Endpoints ---

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generates a Mario level based on the provided attribute targets and parameters.
    """
    start_time = time.time()
    
    try:
        # 1. Map request to internal attributes list [enemies, gaps, pipes]
        attr_list = [
            float(request.attributes.enemies),
            float(request.attributes.gaps),
            float(request.attributes.pipes)
        ]
        
        # 2. Run Mamba Generation
        # Each 'patch' in generate_level is 16 columns
        grid = generate_level(
            attributes=attr_list,
            patches=request.params.num_columns / 16.0,
            seed=request.params.seed,
            temperature=request.params.temperature,
            top_k=request.params.top_k,
            top_p=request.params.top_p,
            cfg_scale=request.params.cfg_scale
        )
        
        # 3. Handle Map Fixing if requested
        final_grid = grid
        is_solvable = False
        fixed_status = False
        rounds_taken = 0
        
        if request.api_key and request.api_key.strip():
            # Run the fix pipeline (it's a generator)
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
            # No API key provided, just check initial solvability
            _, _, is_solvable, _ = bfs_reachability(grid)
        
        # 4. Prepare level string
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
    """Health check endpoint to verify the server is running."""
    return {"status": "ready", "model": "Conditional-Mamba-Mario-v1"}

if __name__ == "__main__":
    print("Starting Mamba Mario API Server...")
    # Run on all interfaces (0.0.0.0) so it's accessible from other devices if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
