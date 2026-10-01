"""Request und Response der API. Pydantic prueft alles, bevor unser Code laeuft."""

from typing import Literal

from pydantic import BaseModel, Field


class Search(BaseModel):
    """Der Suchkontext, er gilt fuer alle Hotels der Anfrage."""

    days_until_checkin: int = Field(ge=0)
    stay_length: int = Field(ge=1)
    device: Literal["mobile", "desktop", "tablet"]
    user_segment: Literal["family", "business", "leisure"]


class Hotel(BaseModel):
    hotel_id: str
    city: str
    stars: int = Field(ge=1, le=5)           # 0 wuerde in price_per_star durch 0 teilen
    rating: float | None = Field(default=None, ge=0, le=10)   # None = neues Hotel
    review_count: int = Field(ge=0)
    price_per_night: float = Field(gt=0)
    distance_to_center_km: float = Field(ge=0)
    has_pool: int = Field(ge=0, le=1)
    breakfast_included: int = Field(ge=0, le=1)
    free_cancellation: int = Field(ge=0, le=1)


class RankRequest(BaseModel):
    search: Search
    hotels: list[Hotel] = Field(min_length=1, max_length=500)


class RankedHotel(BaseModel):
    hotel_id: str
    score: float
    rank: int


class RankResponse(BaseModel):
    model_version: str
    ranking: list[RankedHotel]
