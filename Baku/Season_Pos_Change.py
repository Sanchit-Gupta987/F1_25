import pandas as pd
import fastf1 as ff1
import numpy as np
from typing import Dict

# --- CONFIGURATION ---
ANALYSIS_YEAR = 2025
CURRENT_RACE = 'Azerbaijan Grand Prix'

def get_session_results(year: int, race: str) -> pd.DataFrame:
    """Helper function to get race results."""
    try:
        session = ff1.get_session(year, race, 'R')
        session.load(telemetry=False, laps=False, weather=False, messages=False)
        return session.results
    except Exception as e:
        print(f"Error loading {year} {race}: {e}")
        return pd.DataFrame()

def get_season_position_change(year: int, current_race: str) -> Dict[str, float]:
    """
    Calculates the average position change for each driver for all races
    in the current season *before* the current one.
    """
    print(f"Phase 1: Calculating Season_Position_Change for {year} races before {current_race}...")
    position_changes = {}
    
    schedule = ff1.get_event_schedule(year)
    
    try:
        current_race_index = schedule[schedule['EventName'] == current_race].index[0]
        races_to_analyze = schedule.loc[:current_race_index - 1]
    except IndexError:
        print(f"Warning: Could not find '{current_race}' in the schedule. Using all races up to this point.")
        races_to_analyze = schedule

    for _, race_event in races_to_analyze.iterrows():
        print(f"  - Processing {race_event['EventName']}...")
        race_results = get_session_results(year, race_event['EventName'])
        
        if race_results.empty:
            continue
            
        for _, driver_result in race_results.iterrows():
            driver_code = driver_result['Abbreviation']
            quali_pos = driver_result['GridPosition']
            race_pos = driver_result['Position']
            
            if pd.notna(quali_pos) and pd.notna(race_pos) and ('Finished' in driver_result['Status'] or 'Lap' in driver_result['Status']):
                change = quali_pos - race_pos
                position_changes.setdefault(driver_code, []).append(change)
                
    avg_changes = {driver: np.mean(changes) for driver, changes in position_changes.items() if changes}
    
    print("\nPhase 2: Generating output dictionary...")
    
    # Corrected way to get a list of all drivers for the season.
    all_drivers = set()
    for _, race_event in schedule.iterrows():
        try:
            session = ff1.get_session(year, race_event['EventName'], 'R')
            session.load(telemetry=False, laps=False, weather=False, messages=False)
            all_drivers.update(session.results['Abbreviation'].unique())
        except Exception:
            continue

    final_changes = {}
    for driver in sorted(list(all_drivers)):
        final_changes[driver] = avg_changes.get(driver, 0.0)
    
    print("  - Season position change data generated successfully.")
    return final_changes


def format_for_copy_paste(data: Dict[str, float]):
    """Formats the dictionary into a string ready for copy-pasting."""
    output = "SEASON_POSITION_CHANGE_PREDICTIONS = {\n"
    for driver, value in sorted(data.items()):
        output += f"    '{driver}': {value:.2f},\n"
    output += "}\n"
    return output


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    ff1.Cache.enable_cache('f1_cache')
    season_change_data = get_season_position_change(ANALYSIS_YEAR, CURRENT_RACE)
    
    print("\n" + "="*80)
    print("  COPY AND PASTE THE FOLLOWING INTO YOUR POWER RANKINGS SCRIPT  ")
    print("="*80 + "\n")
    
    formatted_output = format_for_copy_paste(season_change_data)
    print(formatted_output)