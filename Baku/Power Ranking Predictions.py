import fastf1 as ff1
import pandas as pd
from datetime import datetime

# This file contains pre-calculated metrics for various predictive models.

# ==============================================================================
# METRIC 1: HISTORICAL POSITION CHANGE AT BAKU / THIS SEASON
# ==============================================================================
# Predicted position change from qualifying to race finish at the Baku circuit.
# Model: Historical average of finished races (2022-2024), excluding DNFs and major outliers.

BAKU_POSITION_CHANGE_PREDICTIONS = {
    'VER': 1.00,
    'SAI': -1.00,
    'LAW': 0.00,
    'ANT': 2.33,
    'RUS': 2.33,
    'TSU': -3.50,
    'NOR': 3.67,
    'HAD': 0.00,
    'PIA': 0.00,
    'LEC': -1.50,
    'ALO': 2.00,
    'HAM': 4.00,
    'BOR': 2.00,
    'STR': -2.00,
    'BEA': 0.00,
    'COL': 0.00,
    'HUL': 2.00,
    'GAS': 3.33,
    'ALB': 2.33,
    'OCO': 4.00
}

SEASON_POSITION_CHANGE_PREDICTIONS = {
    'ALB': 3.00,
    'ALO': 0.33,
    'ANT': 0.17,
    'BEA': 4.36,
    'BOR': 0.23,
    'COL': 0.11,
    'DOO': 1.50,
    'GAS': 0.15,
    'HAD': -0.50,
    'HAM': 2.21,
    'HUL': 4.57,
    'LAW': 0.92,
    'LEC': -0.07,
    'NOR': 1.07,
    'OCO': 2.60,
    'PIA': -0.25,
    'RUS': -0.19,
    'SAI': 0.15,
    'STR': 2.40,
    'TSU': 0.00,
    'VER': -0.80,
}

# ==============================================================================
# METRIC 2: CURRENT CHAMPIONSHIP STANDINGS
# ==============================================================================
# Data sourced from formula1.com on 2025-09-20.

DRIVER_POINTS = {
    'Oscar Piastri': 324, 'Lando Norris': 293, 'Max Verstappen': 230, 'George Russell': 194,
    'Charles Leclerc': 163, 'Lewis Hamilton': 117, 'Alexander Albon': 70, 'Kimi Antonelli': 66,
    'Isack Hadjar': 38, 'Nico Hulkenberg': 37, 'Lance Stroll': 32, 'Fernando Alonso': 30,
    'Esteban Ocon': 28, 'Pierre Gasly': 20, 'Liam Lawson': 20, 'Gabriel Bortoleto': 18,
    'Oliver Bearman': 16, 'Carlos Sainz': 16, 'Yuki Tsunoda': 12, 'Franco Colapinto': 0,
    'Jack Doohan': 0
}

CONSTRUCTOR_POINTS = {
    'McLaren': 617, 'Ferrari': 280, 'Mercedes': 260, 'Red Bull Racing': 239,
    'Williams': 86, 'Aston Martin': 62, 'Racing Bulls': 61, 'Kick Sauber': 55,
    'Haas': 44, 'Alpine': 20
}

# ==============================================================================
# METRIC 3: BAKU - WEIGHTED QUALIFYING SCORE
# ==============================================================================

BAKU_QUALIFYING_SCORE = {
    'VER': 28.30, 'SAI': 26.89, 'LAW': 25.47, 'ANT': 24.05, 'RUS': 22.64,
    'TSU': 21.23, 'NOR': 19.81, 'HAD': 18.39, 'PIA': 16.98, 'LEC': 15.57,
    'ALO': 14.15, 'HAM': 12.73, 'BOR': 11.32, 'STR': 9.91, 'BEA': 8.49,
    'COL': 7.08, 'HUL': 5.66, 'GAS': 4.25, 'ALB': 2.83, 'OCO': 1.42
}

# ==============================================================================
# METRIC 4: HISTORICAL TRACK CHARACTERISTICS
# ==============================================================================

TRACK_DATA = {
    'Baku': {'pole_rate': 0.38, 'overtake_difficulty': 0.45},
    'Monaco': {'pole_rate': 0.48, 'overtake_difficulty': 0.95},
    'Bahrain': {'pole_rate': 0.35, 'overtake_difficulty': 0.20},
    'Singapore': {'pole_rate': 0.67, 'overtake_difficulty': 0.85},
    'Austin': {'pole_rate': 0.42, 'overtake_difficulty': 0.35},
    'Mexico City': {'pole_rate': 0.40, 'overtake_difficulty': 0.70},
    'São Paulo': {'pole_rate': 0.36, 'overtake_difficulty': 0.25},
    'Las Vegas': {'pole_rate': 0.00, 'overtake_difficulty': 0.30},
    'Qatar': {'pole_rate': 0.50, 'overtake_difficulty': 0.65},
    'Abu Dhabi': {'pole_rate': 0.69, 'overtake_difficulty': 0.55}
}

# --- CONFIGURATION ---
ANALYSIS_YEAR = 2025
ANALYSIS_TRACK = 'Baku'


# --- HELPER FUNCTIONS ---

def get_clean_race_pace(year: int, track: str):
    """
    Calculates the average 'clean air' lap time for each driver from a past race.
    """
    print(f"\nFetching historical race pace from {year} {track}...")
    try:
        session = ff1.get_session(year, track, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        
        # Filter out junk laps
        clean_laps = laps.loc[(laps['PitInTime'].isna()) & (laps['PitOutTime'].isna()) & (laps['IsAccurate'] == True)]
        
        # Calculate average pace and return as a dictionary
        pace = clean_laps.groupby('Driver')['LapTime'].mean().dt.total_seconds()
        print(" -> Done.")
        return pace.to_dict()
    except Exception as e:
        print(f" -> Could not fetch historical pace data. Error: {e}")
        return {}

def calculate_power_score(driver_data: dict, min_pace: float):
    """
    Calculates a final Power Score for a driver by combining all predictive metrics.
    """
    # Unpack the data for the driver
    pos = driver_data['Position']
    driver_code = driver_data['Driver']
    team = driver_data['Team']
    historical_pos_change = driver_data['HistPosChange']
    historical_race_pace = driver_data['HistRacePace']

    # New: Get the season-long position change data
    season_pos_change = driver_data['SeasonPosChange']
    
    # New: Combine historical and season change with a weighted average
    # You can adjust the weights to prioritize one metric over the other.
    # Here, we give more weight to the specific Baku historical data.
    combined_pos_change = (historical_pos_change * 0.7) + (season_pos_change * 0.3)
    
    # Get track data from the imported dictionary
    track_info = TRACK_DATA.get(ANALYSIS_TRACK)
    if not track_info:
        print(f"Warning: No track data for {ANALYSIS_TRACK}, using default weights.")
        track_info = {'pole_rate': 0.4, 'overtake_difficulty': 0.5} # Default values
    
    # --- Calculate Qualifying Component ---
    base_quali_score = 21 - pos
    track_difficulty = (track_info['pole_rate'] + track_info['overtake_difficulty']) / 2
    track_weight = 1 + track_difficulty
    
    max_constructor_points = max(CONSTRUCTOR_POINTS.values()) if CONSTRUCTOR_POINTS else 1
    team_strength = CONSTRUCTOR_POINTS.get(team, 0) / max_constructor_points
    form_weight = 1 + team_strength
    
    qualifying_component = base_quali_score * track_weight * form_weight

    # --- Calculate Racing Component ---
    if historical_race_pace and min_pace > 0:
        pace_score = (min_pace / historical_race_pace) * 20
    else:
        pace_score = 10 
        
    # Updated: The racing component now uses the combined position change
    racing_component = pace_score + combined_pos_change

    # --- Final Score ---
    power_score = qualifying_component + racing_component
    return power_score

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    ff1.Cache.enable_cache('f1_cache')
    print(f"--- Generating Race Prediction for {ANALYSIS_YEAR} {ANALYSIS_TRACK} ---")
    
    try:
        quali_session = ff1.get_session(ANALYSIS_YEAR, ANALYSIS_TRACK, 'Q')
        quali_session.load(laps=False, telemetry=False, weather=False, messages=False)
        quali_results = quali_session.results
        print(f"\nSuccessfully loaded qualifying data for {ANALYSIS_YEAR} {ANALYSIS_TRACK}.")
    except Exception as e:
        print(f"Fatal Error: Could not load qualifying data. Cannot make prediction. \nError: {e}")
        exit()

    historical_pace_data = get_clean_race_pace(ANALYSIS_YEAR - 1, ANALYSIS_TRACK)
    if not historical_pace_data:
        print("Warning: Could not get historical pace data. Predictions will be less accurate.")
        min_pace = 0
    else:
        min_pace = min(historical_pace_data.values())

    print("\nCalculating Power Score for each driver...")
    final_predictions = []
    
    for _, driver in quali_results.iterrows():
        driver_code = driver['Abbreviation']
        
        # --- MODIFICATION START ---
        # Update the dictionary to include the new metric
        driver_data_for_model = {
            'Driver': driver_code,
            'Team': driver['TeamName'],
            'Position': driver['Position'],
            'HistPosChange': BAKU_POSITION_CHANGE_PREDICTIONS.get(driver_code, 0),
            'SeasonPosChange': SEASON_POSITION_CHANGE_PREDICTIONS.get(driver_code, 0), # New metric
            'HistRacePace': historical_pace_data.get(driver_code)
        }
        # --- MODIFICATION END ---

        power_score = calculate_power_score(driver_data_for_model, min_pace)
        
        final_predictions.append({
            'Driver': driver_code,
            'StartingPos': driver['Position'],
            'PowerScore': power_score
        })
        
        print(f" -> {driver_code}: {power_score:.2f}")

    predicted_order = pd.DataFrame(final_predictions)
    predicted_order = predicted_order.sort_values(by='PowerScore', ascending=False).reset_index(drop=True)
    predicted_order['PredictedPos'] = predicted_order.index + 1
    
    print("\n--- 🏁 Final Predicted Race Order 🏁 ---")
    print(predicted_order[['PredictedPos', 'StartingPos', 'Driver', 'PowerScore']].to_string(index=False))