import fastf1 as ff1
import pandas as pd

# Historical track data
TRACK_DATA = {
    'Baku': {'pole_rate': 0.38, 'overtake_difficulty': 0.45},
    'Singapore': {'pole_rate': 0.67, 'overtake_difficulty': 0.85},
    'Austin': {'pole_rate': 0.42, 'overtake_difficulty': 0.35},
    'Mexico City': {'pole_rate': 0.40, 'overtake_difficulty': 0.70},
    'São Paulo': {'pole_rate': 0.36, 'overtake_difficulty': 0.25},
    'Las Vegas': {'pole_rate': 0.00, 'overtake_difficulty': 0.30}, 
    'Qatar': {'pole_rate': 0.50, 'overtake_difficulty': 0.65}, 
    'Abu Dhabi': {'pole_rate': 0.69, 'overtake_difficulty': 0.55}
}

def generate_qualifying_score_dict(year: int, track: str):
    """
    Calculates a weighted qualifying score and formats it as a Python
    dictionary for easy export.
    """
    try:
        # --- 1. Load Qualifying Data ---
        ff1.Cache.enable_cache('f1_cache')
        session = ff1.get_session(year, track, 'Q')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results
        
        print(f"--- Generating Score Dictionary for {session.event['EventName']} {year} ---\n")
        
        # --- 2. Check if Track Data Exists ---
        track_data = TRACK_DATA.get(track)
        if not track_data:
            print(f"Error: No historical data found for '{track}'.")
            return None
            
        track_difficulty = (track_data['pole_rate'] + track_data['overtake_difficulty']) / 2

        # --- 3. Calculate Score for Each Driver ---
        scored_results = []
        for _, driver_data in results.iterrows():
            pos = driver_data['Position']
            base_score = 21 - pos
            final_score = base_score * (1 + track_difficulty)
            
            scored_results.append({
                'Driver': driver_data['Abbreviation'],
                'WeightedScore': final_score
            })
            
        final_table = pd.DataFrame(scored_results)
        
        # --- 4. Convert DataFrame to Dictionary and Print ---
        # Set 'Driver' as the index and select the 'WeightedScore' column
        score_dict = final_table.set_index('Driver')['WeightedScore']
        # Round the scores to two decimal places for cleanliness
        score_dict = score_dict.round(2).to_dict()
        
        print("--- Copy and paste the dictionary below into your predictions.py file ---\n")
        
        # Print the dictionary in a formatted way
        dict_name = f"{track.upper().replace(' ', '_')}_QUALIFYING_SCORE"
        print(f"{dict_name} = {{")
        for driver, score in score_dict.items():
            print(f"    '{driver}': {score},")
        print("}")

        return score_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    # --- Set the year and track you want to generate data for ---
    analysis_year = 2024
    analysis_track = 'Baku'
    
    generate_qualifying_score_dict(analysis_year, analysis_track)