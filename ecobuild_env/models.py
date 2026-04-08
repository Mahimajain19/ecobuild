from pydantic import BaseModel, ConfigDict
from typing import Optional

class BuildingObservation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    indoor_temperature: float  # Current indoor temp (°C)
    outdoor_temperature: float  # Current outdoor temp (°C)
    occupancy: int  # 0 = vacant, 1 = occupied
    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6 (Monday=0)
    heater_status: int  # 0 = off, 1 = on
    lights_status: int  # 0 = off, 1 = on
    time_step: int  # Current step in episode

class BuildingAction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    heater_control: int  # 0 = off, 1 = on
    lights_control: int  # 0 = off, 1 = on

class BuildingReward(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    reward: float
    energy_cost: float
    comfort_penalty: float
    vacancy_waste: float
    info: Optional[dict] = None
