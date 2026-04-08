import numpy as np

def is_occupied_fixed(hour, day_of_week):
    """
    Fixed occupancy schedule: 8am-6pm on weekdays, vacant otherwise.
    """
    if day_of_week >= 5:  # Weekend
        return 0
    return 1 if 8 <= hour < 18 else 0

def is_occupied_stochastic(hour, day_of_week):
    """
    Stochastic occupancy: Probabilistic arrivals/departures.
    """
    if day_of_week >= 5:
        return 1 if np.random.random() < 0.1 else 0  # 10% chance on weekends
    
    if 8 <= hour < 18:
        return 1 if np.random.random() < 0.9 else 0  # 90% chance during work hours
        
    return 1 if np.random.random() < 0.05 else 0  # 5% chance other times
