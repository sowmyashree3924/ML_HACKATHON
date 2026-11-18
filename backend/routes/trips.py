"""HTTP routes for listing, inspecting, and matching student-test trips."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services.data_service import get_student_test_trips, get_trip
from backend.services.prediction_service import match_trip

router = APIRouter(prefix="/api/trips", tags=["Trips"])


@router.get("/")
def list_trips() -> dict[str, object]:
    """Return trip IDs available in the student-test partition."""
    try:
        trips = get_student_test_trips()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "success", "total_trips": len(trips), "trips": trips}


@router.get("/{trip_id}")
def trip_details(trip_id: str) -> dict[str, object]:
    """Return GPS observations and network size for a selected trip."""
    try:
        trip = get_trip(trip_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    points = [
        {"longitude": float(row.longitude), "latitude": float(row.latitude), "timestamp": float(row.timestamp)}
        for row in trip.track.itertuples(index=False)
    ]
    return {
        "trip_id": trip.trip_id,
        "observations": points,
        "node_count": len(trip.nodes),
        "arc_count": len(trip.arcs),
    }


@router.post("/{trip_id}/match")
def predict_trip_route(trip_id: str) -> dict[str, object]:
    """Return predicted matched road segments for a trip."""
    try:
        result = match_trip(trip_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        code = 503 if "checkpoint" in str(error).lower() else 404
        raise HTTPException(status_code=code, detail=str(error)) from error
    return {
        "trip_id": trip_id,
        "matched_observations": len(result["matches"]),
        "route_arc_count": len(result["route"]),
        "matches": result["matches"],
        "route": result["route"],
    }
