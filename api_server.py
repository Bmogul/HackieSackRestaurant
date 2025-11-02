"""
FastAPI HTTP/HTTPS Server for Restaurant Booking MCP Tools.

This exposes the MCP server tools as REST API endpoints, allowing
web clients, mobile apps, and remote AI services to access the
restaurant booking system over HTTP/HTTPS.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import MCP tools
from tools import execute_tool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Restaurant Booking API",
    description="REST API wrapper for MCP Restaurant Booking System with hybrid RAG (ChromaDB + AWS Bedrock)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (configure as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (simple bearer token)
API_KEY = "your-secret-api-key-here"  # Change this in production!

def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Verify API key from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Bearer <token>"
        )

    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return True


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SearchRestaurantsRequest(BaseModel):
    cuisine: Optional[str] = Field(None, description="Cuisine type (Italian, Japanese, etc.)")
    location: Optional[str] = Field(None, description="Location/neighborhood")
    price_range: Optional[str] = Field(None, description="Price range: $, $$, $$$, $$$$")
    dietary_options: Optional[List[str]] = Field(None, description="Dietary options (vegetarian, vegan, etc.)")
    min_rating: Optional[float] = Field(None, ge=0, le=5, description="Minimum rating (0-5)")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Max results")


class CheckAvailabilityRequest(BaseModel):
    restaurant_id: str = Field(..., description="Restaurant ID")
    date: str = Field(..., description="Date in YYYY-MM-DD format", pattern=r"^\d{4}-\d{2}-\d{2}$")
    time: str = Field(..., description="Time in HH:MM format", pattern=r"^\d{2}:\d{2}$")


class QueryKnowledgeBaseRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    limit: Optional[int] = Field(5, ge=1, le=20, description="Max results")


class MakeReservationRequest(BaseModel):
    restaurant_id: str = Field(..., description="Restaurant ID")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    time: str = Field(..., description="Time in HH:MM format")
    party_size: int = Field(..., ge=1, le=20, description="Number of guests (1-20)")
    customer_name: str = Field(..., min_length=1, description="Customer full name")
    customer_phone: str = Field(..., description="Customer phone number")
    special_requests: Optional[str] = Field(None, description="Special requests or dietary requirements")


class ManageBookingRequest(BaseModel):
    action: str = Field(..., description="Action: 'get' or 'cancel'", pattern=r"^(get|cancel)$")
    booking_id: Optional[str] = Field(None, description="Booking confirmation number")


class ToolResponse(BaseModel):
    success: bool
    data: str
    timestamp: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    timestamp: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Call MCP tool and extract text response."""
    try:
        result = await execute_tool(tool_name, arguments)

        # Extract text from result
        text_parts = []
        for content in result:
            if hasattr(content, 'text'):
                text_parts.append(content.text)

        return "\n".join(text_parts) if text_parts else "No response"

    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution error: {str(e)}"
        )


def create_response(data: str) -> ToolResponse:
    """Create successful response."""
    return ToolResponse(
        success=True,
        data=data,
        timestamp=datetime.utcnow().isoformat()
    )


def create_error(error: str) -> ErrorResponse:
    """Create error response."""
    return ErrorResponse(
        success=False,
        error=error,
        timestamp=datetime.utcnow().isoformat()
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Restaurant Booking API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "search": "/api/v1/search",
            "availability": "/api/v1/availability",
            "query": "/api/v1/query",
            "reserve": "/api/v1/reserve",
            "bookings": "/api/v1/bookings"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/search", response_model=ToolResponse, dependencies=[Depends(verify_api_key)])
async def search_restaurants(request: SearchRestaurantsRequest):
    """
    Search and filter restaurants.

    Requires Bearer token authentication.
    """
    logger.info(f"Search request: {request.dict()}")

    arguments = {k: v for k, v in request.dict().items() if v is not None}
    result = await call_mcp_tool("search_restaurants", arguments)

    return create_response(result)


@app.post("/api/v1/availability", response_model=ToolResponse, dependencies=[Depends(verify_api_key)])
async def check_availability(request: CheckAvailabilityRequest):
    """
    Check table availability at specific date/time.

    Requires Bearer token authentication.
    """
    logger.info(f"Availability check: {request.restaurant_id} on {request.date} at {request.time}")

    result = await call_mcp_tool("check_availability", request.dict())

    return create_response(result)


@app.post("/api/v1/query", response_model=ToolResponse, dependencies=[Depends(verify_api_key)])
async def query_knowledge_base(request: QueryKnowledgeBaseRequest):
    """
    Query restaurant knowledge base using hybrid RAG (ChromaDB + AWS Bedrock).

    Performs semantic search across local structured data and cloud documents.

    Requires Bearer token authentication.
    """
    logger.info(f"Knowledge base query: {request.query}")

    result = await call_mcp_tool("query_knowledge_base", request.dict())

    return create_response(result)


@app.post("/api/v1/reserve", response_model=ToolResponse, dependencies=[Depends(verify_api_key)])
async def make_reservation(request: MakeReservationRequest):
    """
    Create a new restaurant reservation.

    Validates availability and creates booking with confirmation number.

    Requires Bearer token authentication.
    """
    logger.info(f"Reservation request: {request.customer_name} at {request.restaurant_id}")

    result = await call_mcp_tool("make_reservation", request.dict())

    return create_response(result)


@app.post("/api/v1/bookings", response_model=ToolResponse, dependencies=[Depends(verify_api_key)])
async def manage_bookings(request: ManageBookingRequest):
    """
    Manage booking state (get or cancel).

    Actions:
    - get: Retrieve booking details by ID
    - cancel: Cancel an existing booking

    Requires Bearer token authentication.
    """
    logger.info(f"Booking management: {request.action} - {request.booking_id}")

    result = await call_mcp_tool("manage_booking_state", request.dict())

    return create_response(result)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error(exc.detail).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error(str(exc)).dict()
    )


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on server startup."""
    logger.info("=" * 70)
    logger.info("Restaurant Booking API Server Starting")
    logger.info("=" * 70)
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Health Check: http://localhost:8000/health")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on server shutdown."""
    logger.info("Restaurant Booking API Server Shutting Down")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Restaurant Booking API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--ssl-keyfile", help="Path to SSL key file for HTTPS")
    parser.add_argument("--ssl-certfile", help="Path to SSL certificate file for HTTPS")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Run server
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        reload=args.reload,
        log_level="info"
    )
