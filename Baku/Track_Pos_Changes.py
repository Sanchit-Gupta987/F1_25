import fastf1 as ff1
import pandas as pd
from datetime import datetime

def get_historical_averages(track, start_year, end_year):
    """
    Analyzes past races at a track to build a historical average position
    change for each driver, EXCLUDING DNFs and major outliers.
    """
    print("Phase 1: Analyzing historical data (excluding DNFs and outliers)...")
    driver_stats = {}

    for year in range(start_year, end_year + 1):
        try:
            session = ff1.get_session(year, track, 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            print(f"  - Analyzing {session.event['EventName']} {year}...")
            
            for _, driver_result in session.results.iterrows():
                status = driver_result['Status']
                
                if 'Finished' in status or 'Lap' in status:
                    driver_code = driver_result['Abbreviation']
                    driver_stats.setdefault(driver_code, [])
                    
                    quali_pos = driver_result['GridPosition']
                    race_pos = driver_result['Position']
                    
                    if pd.notna(quali_pos) and pd.notna(race_pos) and quali_pos > 0:
                        position_change = quali_pos - race_pos
                        
                        # FINAL IMPROVEMENT: Exclude extreme negative outliers
                        if position_change > -10:
                            driver_stats[driver_code].append(position_change)

        except Exception as e:
            print(f"  - Could not process data for {track} in {year}. Error: {e}")

    historical_averages = {}
    for driver, changes in driver_stats.items():
        if changes:
            historical_averages[driver] = sum(changes) / len(changes)
            
    print("  - Historical analysis complete.")
    return historical_averages


def generate_predictions(track: str):
    """
    Generates race predictions based on current qualifying results and
    historical performance data.
    """
    ff1.Cache.enable_cache('f1_cache')
    current_year = datetime.now().year
    
    historical_data = get_historical_averages(track, 2022, current_year - 1)
    
    print("\nPhase 2: Fetching current qualifying data...")
    try:
        quali = ff1.get_session(current_year, track, 'Q')
        quali.load()
        quali_results = quali.results
        print(f"  - Successfully loaded qualifying data for {quali.event['EventName']} {current_year}.")
    except Exception as e:
        print(f"Error: Could not load qualifying data. Has the session finished? \n({e})")
        return

    print("\nPhase 3: Generating predictions...")
    teammate_map = {}
    for team in quali_results['TeamName'].unique():
        teammates = quali_results[quali_results['TeamName'] == team]['Abbreviation'].tolist()
        if len(teammates) == 2:
            teammate_map[teammates[0]] = teammates[1]
            teammate_map[teammates[1]] = teammates[0]

    predictions = []
    for _, driver_data in quali_results.iterrows():
        driver = driver_data['Abbreviation']
        note = "Historical Avg"
        
        if driver in historical_data:
            predicted_change = historical_data[driver]
        else:
            teammate = teammate_map.get(driver)
            if teammate and teammate in historical_data:
                predicted_change = historical_data[teammate]
                note = f"Teammate Avg ({teammate})"
            else:
                predicted_change = 0.0
                note = "No Data"

        predictions.append({
            'Quali Pos': int(driver_data['Position']),
            'Driver': driver,
            'Team': driver_data['TeamName'],
            'Predicted Change': predicted_change,
            'Note': note
        })

    print("\n--- Baku Race Prediction Data (DNFs & Outliers Excluded) ---")
    print(f"{'Pos':<4} {'Driver':<10} {'Predicted Change':<18} {'Note'}")
    print("-" * 55)

    sorted_preds = sorted(predictions, key=lambda x: x['Quali Pos'])
    for p in sorted_preds:
        print(f"{p['Quali Pos']:<4} {p['Driver']:<10} {p['Predicted Change']:>+7.2f}{' ':<11} {p['Note']}")


if __name__ == '__main__':
    generate_predictions('Baku')