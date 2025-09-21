import pandas as pd
import xgboost as xgb
import fastf1 as ff1
from typing import List, Dict, Any
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np

# --- CONFIGURATION ---
ff1.Cache.enable_cache('f1_cache')
PREDICTION_YEAR = 2025
PREDICTION_RACE = 'Azerbaijan Grand Prix'
TARGET_TRACK_CSV_NAME = 'Azerbaijan Grand Prix'

# --- DATA MAPPING AND CONSTANTS ---
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

POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

# --- DATA FETCHING FUNCTIONS ---
def get_session_results(year: int, race: str, session_type: str) -> pd.DataFrame:
    try:
        session = ff1.get_session(year, race, session_type)
        session.load(telemetry=False, laps=False, weather=False, messages=False)
        return session.results
    except Exception as e:
        print(f"Error loading {year} {race} {session_type}: {e}")
        return pd.DataFrame()

def get_clean_race_pace(year: int, track: str, session_type: str) -> Dict[str, float]:
    try:
        session = ff1.get_session(year, track, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        clean_laps = laps.loc[(laps['PitInTime'].isna()) & (laps['PitOutTime'].isna()) & (laps['IsAccurate'] == True)]
        pace = clean_laps.groupby('Driver')['LapTime'].mean().dt.total_seconds()
        return pace.to_dict()
    except Exception as e:
        return {}

def get_weighted_quali_score(quali_results: pd.DataFrame, track: str) -> Dict[str, float]:
    track_name_key = next((k for k, v in TRACK_DATA.items() if v.get('pole_rate') == TRACK_DATA['Baku']['pole_rate']), 'Baku')
    track_data = TRACK_DATA.get(track_name_key, {'pole_rate': 0.5, 'overtake_difficulty': 0.5})
    track_difficulty = (track_data['pole_rate'] + track_data['overtake_difficulty']) / 2
    score_dict = {}
    for _, driver_data in quali_results.iterrows():
        pos = driver_data['Position']
        base_score = 21 - pos
        final_score = (base_score * (1 + track_difficulty) ) ** 1.2
        score_dict[driver_data['Abbreviation']] = round(final_score, 2)
    return score_dict

def get_driver_and_team_points(year: int, races_before: List[str]) -> tuple[Dict[str, float], Dict[str, float]]:
    driver_points = {}
    team_points = {}
    for race in races_before:
        race_results = get_session_results(year, race, 'R')
        for _, driver_data in race_results.iterrows():
            pos = driver_data['Position']
            driver_code = driver_data['Abbreviation']
            team_name = driver_data['TeamName']
            if pos in POINTS_SYSTEM:
                points_to_add = POINTS_SYSTEM[pos]
                driver_points[driver_code] = driver_points.get(driver_code, 0) + points_to_add
                team_points[team_name] = team_points.get(team_name, 0) + points_to_add
    return driver_points, team_points

def get_historical_position_change(year: int, track: str) -> Dict[str, float]:
    driver_stats = {}
    for prev_year in range(2022, year):
        try:
            race_results = get_session_results(prev_year, track, 'R')
            for _, driver_result in race_results.iterrows():
                status = driver_result['Status']
                if 'Finished' in status or 'Lap' in status:
                    driver_code = driver_result['Abbreviation']
                    quali_pos = driver_result['GridPosition']
                    race_pos = driver_result['Position']
                    if pd.notna(quali_pos) and pd.notna(race_pos) and quali_pos > 0:
                        position_change = quali_pos - race_pos
                        driver_stats.setdefault(driver_code, []).append(position_change)
        except Exception as e:
            pass
    historical_averages = {driver: sum(changes) / len(changes) for driver, changes in driver_stats.items() if changes}
    return historical_averages

def get_weather_conditions(year: int, race: str, session_type: str) -> Dict[str, float]:
    try:
        session = ff1.get_session(year, race, session_type)
        session.load(telemetry=False, laps=False, weather=True, messages=False)
        weather_data = session.weather_data
        weather_dict = {
            'AirTemp_Avg': weather_data['AirTemp'].mean(),
            'TrackTemp_Avg': weather_data['TrackTemp'].mean(),
            'Humidity_Avg': weather_data['Humidity'].mean(),
            'WindSpeed_Avg': weather_data['WindSpeed'].mean()
        }
        return weather_dict
    except Exception as e:
        return {}
        
def get_car_performance_score(year: int, race: str, session_type: str) -> Dict[str, float]:
    try:
        session = ff1.get_session(year, race, session_type)
        session.load(telemetry=False, laps=True, weather=False, messages=False)
        laps = session.laps
        fastest_lap_time = min(laps['LapTime'].dropna()).total_seconds()
        performance_scores = {}
        for driver_code in laps['Driver'].unique():
            driver_best_lap_time = min(laps.loc[laps['Driver'] == driver_code, 'LapTime']).total_seconds()
            score = (fastest_lap_time / driver_best_lap_time) * 100
            performance_scores[driver_code] = score
        return performance_scores
    except Exception as e:
        return {}

def get_tire_degradation(year: int, race: str, session_type: str) -> Dict[str, float]:
    try:
        session = ff1.get_session(year, race, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        degradation_data = {}
        for driver_code in laps['Driver'].unique():
            driver_laps = laps.loc[laps['Driver'] == driver_code]
            stints = driver_laps.groupby('Stint')['LapNumber'].agg(['min', 'max'])
            stint_degradations = []
            for _, stint_info in stints.iterrows():
                start_lap = stint_info['min']
                end_lap = stint_info['max']
                if end_lap > start_lap + 2:
                    try:
                        start_lap_time = driver_laps.loc[(driver_laps['LapNumber'] == start_lap) & (driver_laps['Driver'] == driver_code)]['LapTime'].iloc[0].total_seconds()
                        end_lap_time = driver_laps.loc[(driver_laps['LapNumber'] == end_lap) & (driver_laps['Driver'] == driver_code)]['LapTime'].iloc[0].total_seconds()
                        degradation = (end_lap_time - start_lap_time) / (end_lap - start_lap)
                        stint_degradations.append(degradation)
                    except (IndexError, AttributeError):
                        continue
            if stint_degradations:
                degradation_data[driver_code] = sum(stint_degradations) / len(stint_degradations)
            else:
                degradation_data[driver_code] = 0.0
        return degradation_data
    except Exception as e:
        return {}

# --- Helper Function for Data Inspection ---
def check_missing_data(df: pd.DataFrame, title: str):
    """Prints a summary of missing data for a given DataFrame."""
    print(f"\n--- Missing Data Check: {title} ---")
    
    # Calculate the number and percentage of missing values per column
    missing_count = df.isnull().sum()
    missing_percent = 100 * missing_count / len(df)
    
    # Create a new DataFrame to display the results
    missing_data_summary = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percent': missing_percent.round(2)
    })
    
    # Filter to show only columns with missing data
    missing_data_summary = missing_data_summary[missing_data_summary['Missing Count'] > 0]
    
    if missing_data_summary.empty:
        print("✅ No missing data found.")
    else:
        print(missing_data_summary.sort_values(by='Missing Count', ascending=False))

# --- MAIN PREDICTION SCRIPT ---
def run_prediction():
    # Load and Prepare Data
    print("--- 1. Loading and Preparing Data ---")
    try:
        master_df = pd.read_csv('f1_master_dataset_final.csv')
        training_data = master_df[master_df['Track'] == TARGET_TRACK_CSV_NAME].copy()
    except FileNotFoundError:
        print("Error: 'f1_master_dataset_final.csv' not found. Please run the data update script first.")
        return

    # Define the full feature list for training
    training_features = [
        'WeightedQualifyingScore',
        'DriverForm',
        'TeamStrength',
        'HistoricalPositionChange',
        'RacePaceScore',
        'TireDegradation',
        'CarPerformanceScore',
        'AirTemp_Avg',
        'TrackTemp_Avg',
        'Humidity_Avg',
        'WindSpeed_Avg'
    ]
    target = 'FinalRacePosition'
    
    # Separate features and target from historical data
    X_train = training_data[training_features]
    y_train = training_data[target]

    # Check for missing data in training set before imputation
    check_missing_data(X_train, "Training Data (before imputation)")

    # Pre-process training data by filling missing values
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

    # Model Training
    print("\n--- 2. Training the XGBoost Model with Tuned Parameters ---")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        gamma=0.2,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training complete. ✅")

    # Generating Predictions
    print(f"\n--- 3. Generating Predictions for {PREDICTION_RACE} {PREDICTION_YEAR} ---")
    try:
        quali_results = get_session_results(PREDICTION_YEAR, PREDICTION_RACE, 'Q')
        if quali_results.empty:
            print(f"Error: Could not load qualifying results for {PREDICTION_RACE} {PREDICTION_YEAR}.")
            return

        races_this_year = [event['EventName'] for _, event in ff1.get_event_schedule(PREDICTION_YEAR).iterrows() if event['EventName'] != 'Test']
        races_until_current = races_this_year[:races_this_year.index(PREDICTION_RACE)]
        
        # Call the helper functions to get the features for the current race
        quali_scores = get_weighted_quali_score(quali_results, 'Baku')
        driver_points, team_points = get_driver_and_team_points(PREDICTION_YEAR, races_until_current)
        historical_changes = get_historical_position_change(PREDICTION_YEAR, PREDICTION_RACE)
        
        # --- NEW: Get data from current race weekend's practice sessions ---
        # Note: If FP2 or Q are missing, the functions will return an empty dict,
        # which will be handled by imputation.
        race_pace = get_clean_race_pace(PREDICTION_YEAR, PREDICTION_RACE, 'FP2')
        weather_conditions = get_weather_conditions(PREDICTION_YEAR, PREDICTION_RACE, 'FP2')

        prediction_rows = []
        for _, driver_data in quali_results.iterrows():
            driver_code = driver_data['Abbreviation']
            team_name = driver_data['TeamName']
            
            row = {
                'Driver': driver_code,
                'Team': team_name,
                'QualiPosition': driver_data['Position'],
                'WeightedQualifyingScore': quali_scores.get(driver_code, np.nan),
                'DriverForm': driver_points.get(driver_code, np.nan),
                'TeamStrength': team_points.get(team_name, np.nan),
                'HistoricalPositionChange': historical_changes.get(driver_code, np.nan),
                'RacePaceScore': race_pace.get(driver_code, np.nan),
                'TireDegradation': np.nan,  # Data not consistently available
                'CarPerformanceScore': np.nan,  # Data not consistently available
                'AirTemp_Avg': weather_conditions.get('AirTemp_Avg', np.nan),
                'TrackTemp_Avg': weather_conditions.get('TrackTemp_Avg', np.nan),
                'Humidity_Avg': weather_conditions.get('Humidity_Avg', np.nan),
                'WindSpeed_Avg': weather_conditions.get('WindSpeed_Avg', np.nan)
            }
            prediction_rows.append(row)
        
        X_predict = pd.DataFrame(prediction_rows)

        # Check for missing data in prediction set before imputation
        check_missing_data(X_predict, f"Prediction Data ({PREDICTION_RACE}) before imputation")

        # Define the subset of features that are available for prediction
        # based on the missing data check above.
        prediction_features = [
            'WeightedQualifyingScore',
            'DriverForm',
            'TeamStrength',
            'HistoricalPositionChange',
            'RacePaceScore',
            'AirTemp_Avg',
            'TrackTemp_Avg',
            'Humidity_Avg',
            'WindSpeed_Avg'
        ]

        # Filter the training data to match the prediction features.
        # This is the key change to solve the dimensionality mismatch.
        X_train_filtered = X_train[prediction_features]
        
        X_predict_features = X_predict[prediction_features]
        
        # Pre-process prediction data by filling missing values
        imputer = SimpleImputer(strategy='median')
        X_predict_features = pd.DataFrame(imputer.fit_transform(X_predict_features), columns=X_predict_features.columns)

        # Re-train the model on the filtered training data
        model_filtered = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            gamma=0.2,
            random_state=42
        )
        model_filtered.fit(X_train_filtered, y_train)

        predictions = model_filtered.predict(X_predict_features)
        
        # Add the predictions to the original X_predict DataFrame
        X_predict['PredictedFinalPosition'] = predictions.round(0).astype(int)
        
        # Now sort the full DataFrame and re-rank the predictions
        final_ranking = X_predict.sort_values(by='PredictedFinalPosition', ascending=True).reset_index(drop=True)
        final_ranking['PredictedFinalPosition'] = final_ranking.index + 1
        
        print("\n--- 4. Final Predictions ---")
        print(final_ranking[['PredictedFinalPosition', 'QualiPosition', 'Driver', 'Team', 'WeightedQualifyingScore']])
        
        # Plot importance for the filtered model
        plot_importance(model_filtered, importance_type='gain')
        plt.title('Feature Importance (Gain)')
        plt.show()

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    run_prediction()