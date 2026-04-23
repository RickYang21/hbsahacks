"""Environment configuration for the therapy package."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    bluebubbles_url: str = "http://localhost:1234"
    bluebubbles_password: str = ""
    supabase_url: str
    # Accept either SUPABASE_KEY or SUPABASE_SERVICE_KEY (Person A uses the latter).
    supabase_key: str = ""
    supabase_service_key: str = ""
    anthropic_api_key: str
    # For the hackathon we only have one grandma — her phone goes here.
    grandma_phone: str = "+15550000001"

    @property
    def effective_supabase_key(self) -> str:
        return self.supabase_key or self.supabase_service_key


settings = Settings()  # type: ignore[call-arg]
