import numpy as np

def get_outdoor_temp(hour, day, base_temp=15.0, amplitude=8.0):
    """
    Simulate outdoor temperature with a sinusoidal day/night cycle.
    
    Args:
        hour (int): Hour of the day (0-23)
        day (int): Day of the year (0-365)
        base_temp (float): Average daily temperature
        amplitude (float): Temperature variation amplitude
        
    Returns:
        float: Outdoor temperature (°C)
    """
    # Daily cycle: peak at 2pm (14:00), minimum at 2am (02:00)
    daily = amplitude * np.sin(2 * np.pi * (hour - 8) / 24)
    
    # Seasonal variation (slight change over days)
    seasonal = 3.0 * np.sin(2 * np.pi * day / 365)
    
    temp = base_temp + daily + seasonal
    return float(round(temp, 2))
