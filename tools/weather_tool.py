from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

import requests

from agents.schemas import ToolResult
from tools.base import BaseTool
from utils.retry import with_retry, RetryableError



_OPEN_METEO_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherTool(BaseTool):
    """
    Current weather by city.

    Strategy:
    - If OPENWEATHER_API_KEY is present, try OpenWeather first.
    - If OpenWeather fails (bad key, rate limit, network, parsing, etc.), fall back to Open-Meteo (no key).
    - Always returns the same normalized output fields your app expects.
    """

    name = "weather_current"

    def __init__(self, timeout_s: float = 20.0) -> None:
        load_dotenv(find_dotenv()) 
        self.timeout_s = timeout_s
        self.openweather_key = (os.getenv("OPENWEATHER_API_KEY") or "").strip()

    @with_retry(attempts=3)
    def _get(self, url: str, params: Dict[str, Any]) -> requests.Response:
        try:
            r = requests.get(url, params=params, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise RetryableError(f"HTTP request failed: {e}") from e

        # Retry transient errors
        if r.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"Transient HTTP {r.status_code}: {r.text[:200]}")

        return r

    
    def _ow_geocode(self, city: str) -> Optional[Dict[str, Any]]:
        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {"q": city, "limit": 1, "appid": self.openweather_key}
        r = self._get(url, params=params)
        if not r.ok:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        return data[0]

    def _ow_local_time(self, dt_utc: int, tz_offset_s: int) -> str:
        local_dt = datetime.utcfromtimestamp(dt_utc + tz_offset_s)
        return local_dt.strftime("%Y-%m-%dT%H:%M")

    def _openweather_current(self, city: str) -> ToolResult:
        if not self.openweather_key:
            return ToolResult(ok=False, tool_name=self.name, error="OPENWEATHER_API_KEY not set", meta={"provider": "openweather"})

        geo = self._ow_geocode(city)
        if not geo:
            return ToolResult(ok=False, tool_name=self.name, error=f"OpenWeather geocode failed for: {city}", meta={"provider": "openweather"})

        lat = geo.get("lat")
        lon = geo.get("lon")
        resolved_name = geo.get("name") or city
        country = geo.get("country")
        region = geo.get("state")

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": self.openweather_key, "units": "metric"}
        r = self._get(url, params=params)

        if not r.ok:
            # If key is invalid (401) or any other error, we'll fall back.
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"OpenWeather error {r.status_code}: {r.text[:200]}",
                meta={"provider": "openweather", "status_code": r.status_code},
            )

        j = r.json()
        main = j.get("main") or {}
        wind = j.get("wind") or {}
        weather_arr = j.get("weather") or []
        weather0 = weather_arr[0] if weather_arr else {}

        temp_c = main.get("temp")
        feels_c = main.get("feels_like")
        humidity = main.get("humidity")
        wind_ms = wind.get("speed")  # m/s
        wind_kph = (wind_ms * 3.6) if isinstance(wind_ms, (int, float)) else None

        desc = (weather0.get("description") or "").strip()
        conditions = desc[:1].upper() + desc[1:] if desc else None

        dt_utc = j.get("dt")
        tz_offset = j.get("timezone", 0)
        observed_at = self._ow_local_time(int(dt_utc), int(tz_offset)) if isinstance(dt_utc, int) else None

        out = {
            "city": resolved_name,
            "region": region,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "temperature_c": temp_c,
            "apparent_temperature_c": feels_c,
            "humidity_pct": humidity,
            "wind_kph": wind_kph,
            "conditions": conditions,
            "observed_at": observed_at,
        }

        return ToolResult(ok=True, tool_name=self.name, data=out, meta={"provider": "openweather", "status_code": r.status_code})

    
    def _om_geocode(self, city: str) -> Optional[Dict[str, Any]]:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city, "count": 1, "language": "en", "format": "json"}
        r = self._get(url, params=params)
        if not r.ok:
            return None
        j = r.json()
        results = j.get("results") or []
        if not results:
            return None
        return results[0]

    def _open_meteo_current(self, city: str) -> ToolResult:
        geo = self._om_geocode(city)
        if not geo:
            return ToolResult(ok=False, tool_name=self.name, error=f"Open-Meteo geocode failed for: {city}", meta={"provider": "open-meteo"})

        lat = geo.get("latitude")
        lon = geo.get("longitude")
        resolved_name = geo.get("name") or city
        country = geo.get("country")
        region = geo.get("admin1")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": "auto",
        }
        r = self._get(url, params=params)
        if not r.ok:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"Open-Meteo error {r.status_code}: {r.text[:200]}",
                meta={"provider": "open-meteo", "status_code": r.status_code},
            )

        j = r.json()
        cur = j.get("current") or {}

        temp_c = cur.get("temperature_2m")
        feels_c = cur.get("apparent_temperature")
        humidity = cur.get("relative_humidity_2m")
        wind_kph = cur.get("wind_speed_10m")
        code = cur.get("weather_code")
        conditions = _OPEN_METEO_CODE_MAP.get(code, f"Weather code {code}" if code is not None else None)
        observed_at = cur.get("time")

        out = {
            "city": resolved_name,
            "region": region,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "temperature_c": temp_c,
            "apparent_temperature_c": feels_c,
            "humidity_pct": humidity,
            "wind_kph": wind_kph,
            "conditions": conditions,
            "observed_at": observed_at,
        }

        return ToolResult(ok=True, tool_name=self.name, data=out, meta={"provider": "open-meteo", "status_code": r.status_code})

    
    def call(self, tool_args: Dict[str, Any]) -> ToolResult:
        city = str(tool_args.get("city", "")).strip()
        if not city:
            return ToolResult(ok=False, tool_name=self.name, error="Missing 'city'")

        ow_error: Optional[str] = None

        # Try OpenWeather first (only if key exists)
        if self.openweather_key:
            ow_res = self._openweather_current(city)
            if ow_res.ok:
                return ow_res
            ow_error = ow_res.error or "OpenWeather failed"

        # Fall back to Open-Meteo (no key)
        om_res = self._open_meteo_current(city)
        if om_res.ok:
            # annotate that we fell back
            meta = dict(om_res.meta or {})
            if ow_error:
                meta["fallback_from"] = "openweather"
                meta["openweather_error"] = ow_error
            om_res.meta = meta
            return om_res

        # If both fail, return a combined error
        combined = "Weather lookup failed."
        details = []
        if ow_error:
            details.append(f"OpenWeather: {ow_error}")
        details.append(f"Open-Meteo: {om_res.error or 'failed'}")
        return ToolResult(ok=False, tool_name=self.name, error=combined + " " + " | ".join(details), meta={"provider": "combined"})
