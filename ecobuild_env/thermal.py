def update_temperature(current_temp, outdoor_temp, heater_on, alpha=0.95, k=2.5):
    """
    Update indoor temperature using simple RC (Resistor-Capacitor) thermal model.
    
    Args:
        current_temp (float): Current indoor temperature (°C)
        outdoor_temp (float): Current outdoor temperature (°C)
        heater_on (int): 1 if heater is on, 0 otherwise
        alpha (float): Thermal inertia (0-1), higher = slower temperature change
        k (float): Heating effect coefficient (kW to °C/step equivalent)
        
    Returns:
        float: Next indoor temperature (°C)
    """
    heater_power = k if heater_on else 0
    # Next temp is a weighted average of current temp and (outdoor + heater)
    # Simple model: T_{t+1} = alpha * T_t + (1 - alpha) * (T_out + Q_heat)
    next_temp = (
        alpha * current_temp + 
        (1 - alpha) * (outdoor_temp + heater_power)
    )
    return float(round(next_temp, 2))

def calculate_energy_consumption(heater_on, lights_on, heater_power=5.0, lights_power=0.5):
    """
    Calculate energy consumption for the current step in kWh.
    
    Args:
        heater_on (int): 1 if heater is on
        lights_on (int): 1 if lights are on
        heater_power (float): Power rating of the heater in kW
        lights_power (float): Power rating of the lights in kW
        
    Returns:
        float: Energy used in the step
    """
    # Assuming 1 step = 10 minutes (1/6 hour) for energy calculation if needed, 
    # but here we return raw kW/step or kWh equivalent.
    return float(heater_on * heater_power + lights_on * lights_power)
