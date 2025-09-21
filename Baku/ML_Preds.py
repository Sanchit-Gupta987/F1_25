import fastf1 as ff1
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Track characteristics data
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

class F1MLPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.driver_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.track_encoder = LabelEncoder()
        self.best_model = None
        self.feature_names = []
        
    def get_historical_data(self, start_year: int, end_year: int, tracks: List[str]) -> pd.DataFrame:
        """
        Collect historical F1 data for training the ML model.
        Based on the working FastF1 patterns from the provided code.
        """
        print(f"Collecting historical data from {start_year} to {end_year}...")
        all_data = []
        
        # Use the track names as they appear in FastF1, similar to your code
        track_mapping = {
            'Baku': ['Azerbaijan Grand Prix', 'Baku'],
            'Singapore': ['Singapore Grand Prix', 'Singapore'],
            'Austin': ['United States Grand Prix', 'Austin'],
            'Mexico City': ['Mexico City Grand Prix', 'Mexico City'],
            'São Paulo': ['Brazilian Grand Prix', 'São Paulo'],
            'Las Vegas': ['Las Vegas Grand Prix', 'Las Vegas'],
            'Qatar': ['Qatar Grand Prix', 'Qatar'],
            'Abu Dhabi': ['Abu Dhabi Grand Prix', 'Abu Dhabi']
        }
        
        for year in range(start_year, end_year + 1):
            print(f"  Processing {year}...")
            
            # Try each track we have data for
            for track_key in tracks:
                if track_key not in TRACK_DATA:
                    continue
                    
                # Try different track name variations
                track_names_to_try = track_mapping.get(track_key, [track_key])
                
                for track_name in track_names_to_try:
                    try:
                        # Get qualifying and race sessions - following your pattern
                        quali_session = ff1.get_session(year, track_name, 'Q')
                        race_session = ff1.get_session(year, track_name, 'R')
                        
                        quali_session.load(laps=False, telemetry=False, weather=False, messages=False)
                        race_session.load(laps=False, telemetry=False, weather=False, messages=False)
                        
                        # Get season standings before this race
                        season_standings = self.get_season_standings_before_race(year, track_name)
                        
                        # Get historical performance for each driver at this track
                        historical_performance = self.get_historical_performance(year, track_key)
                        
                        # Get race pace data (try to load laps if needed)
                        try:
                            race_session.load(laps=True)
                            race_pace_data = self.get_race_pace_data(race_session)
                        except Exception:
                            race_pace_data = {}
                        
                        # Combine all data
                        race_data = self.combine_race_data(
                            quali_session.results, 
                            race_session.results,
                            season_standings,
                            historical_performance,
                            race_pace_data,
                            track_key,  # Use standardized track name
                            year
                        )
                        
                        all_data.extend(race_data)
                        print(f"    ✓ {track_key} ({track_name}) {year} - {len(race_data)} records")
                        break  # Success, move to next track
                        
                    except Exception as e:
                        print(f"    ✗ {track_key} ({track_name}) {year}: {e}")
                        continue
        
        df = pd.DataFrame(all_data)
        print(f"Collected {len(df)} driver-race records total")
        return df
    
    def get_season_standings_before_race(self, year: int, current_race: str) -> Dict[str, Dict]:
        """Get championship standings before the current race - following your Season_Pos_Change.py pattern."""
        driver_points = {}
        constructor_points = {}
        
        try:
            schedule = ff1.get_event_schedule(year)
            
            # Find current race index - handle case where race name might not match exactly
            current_race_index = None
            for idx, row in schedule.iterrows():
                if current_race in row['EventName'] or row['EventName'] in current_race:
                    current_race_index = idx
                    break
            
            if current_race_index is None:
                print(f"    Warning: Could not find '{current_race}' in schedule")
                return {'driver_points': {}, 'constructor_points': {}}
                
            # Sum points from all previous races - similar to your pattern
            for _, race_event in schedule.loc[:current_race_index - 1].iterrows():
                try:
                    session = ff1.get_session(year, race_event['EventName'], 'R')
                    session.load(telemetry=False, laps=False, weather=False, messages=False)
                    
                    for _, driver_result in session.results.iterrows():
                        driver = driver_result['Abbreviation']
                        team = driver_result['TeamName']
                        points = driver_result['Points'] if pd.notna(driver_result['Points']) else 0
                        
                        driver_points[driver] = driver_points.get(driver, 0) + points
                        constructor_points[team] = constructor_points.get(team, 0) + points
                        
                except Exception as e:
                    print(f"    Could not load points for {race_event['EventName']}: {e}")
                    continue
                    
        except Exception as e:
            print(f"    Error getting season standings: {e}")
            
        return {'driver_points': driver_points, 'constructor_points': constructor_points}
    
    def get_historical_performance(self, year: int, track: str) -> Dict[str, float]:
        """Get historical performance at this track - following your Track_Pos_Changes.py pattern."""
        driver_stats = {}
        
        # Look back 3 years like in your code
        for hist_year in range(max(2022, year-3), year):
            try:
                session = ff1.get_session(hist_year, track, 'R')
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                
                for _, driver_result in session.results.iterrows():
                    status = driver_result['Status']
                    
                    # Only include finished races, following your logic
                    if 'Finished' in status or 'Lap' in status:
                        driver_code = driver_result['Abbreviation']
                        quali_pos = driver_result['GridPosition']
                        race_pos = driver_result['Position']
                        
                        if pd.notna(quali_pos) and pd.notna(race_pos) and quali_pos > 0:
                            position_change = quali_pos - race_pos
                            
                            # Exclude extreme negative outliers like in your code
                            if position_change > -10:
                                if driver_code not in driver_stats:
                                    driver_stats[driver_code] = []
                                driver_stats[driver_code].append(position_change)
                        
            except Exception as e:
                print(f"    Could not process {track} {hist_year}: {e}")
                continue
        
        # Calculate averages
        avg_performance = {}
        for driver, changes in driver_stats.items():
            if changes:
                avg_performance[driver] = sum(changes) / len(changes)
                
        return avg_performance
    
    def get_race_pace_data(self, race_session) -> Dict[str, float]:
        """Extract clean race pace data - following your Power Ranking Predictions pattern."""
        try:
            # Filter out junk laps like in your get_clean_race_pace function
            clean_laps = race_session.laps.loc[
                (race_session.laps['PitInTime'].isna()) & 
                (race_session.laps['PitOutTime'].isna()) & 
                (race_session.laps['IsAccurate'] == True)
            ]
            
            if not clean_laps.empty:
                # Calculate average pace and return as a dictionary, like your code
                pace = clean_laps.groupby('Driver')['LapTime'].mean().dt.total_seconds()
                return pace.to_dict()
        except Exception as e:
            print(f"    Could not fetch pace data: {e}")
        
        return {}
    
    def combine_race_data(self, quali_results, race_results, season_standings, 
                         historical_performance, race_pace_data, track_name, year):
        """Combine all data sources into training examples - following your data combination patterns."""
        combined_data = []
        
        # Get track info like in your power rankings
        track_info = TRACK_DATA.get(track_name, {'pole_rate': 0.4, 'overtake_difficulty': 0.5})
        
        # Process each driver that qualified
        for _, quali_row in quali_results.iterrows():
            driver = quali_row['Abbreviation']
            team = quali_row['TeamName']
            
            # Find corresponding race result
            race_row = race_results[race_results['Abbreviation'] == driver]
            
            if race_row.empty:
                continue
                
            race_row = race_row.iloc[0]
            
            # Skip if race wasn't finished properly (following your DNF exclusion logic)
            status = race_row['Status']
            if pd.isna(race_row['Position']) or not ('Finished' in str(status) or 'Lap' in str(status)):
                continue
                
            # Feature engineering - similar to your power score calculation
            grid_pos = quali_row['GridPosition'] if pd.notna(quali_row['GridPosition']) else quali_row['Position']
            race_pos = race_row['Position']
            
            features = {
                'driver': driver,
                'team': team,
                'track': track_name,
                'year': year,
                'quali_position': quali_row['Position'],
                'grid_position': grid_pos,
                'driver_points_before': season_standings['driver_points'].get(driver, 0),
                'constructor_points_before': season_standings['constructor_points'].get(team, 0),
                'historical_pos_change': historical_performance.get(driver, 0.0),
                'track_pole_rate': track_info['pole_rate'],
                'track_overtake_difficulty': track_info['overtake_difficulty'],
                'track_difficulty': (track_info['pole_rate'] + track_info['overtake_difficulty']) / 2,
                'race_pace': race_pace_data.get(driver, 0.0),
                'target_position': race_pos,
                'position_change': grid_pos - race_pos if pd.notna(grid_pos) else 0
            }
            
            combined_data.append(features)
            
        return combined_data
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """Prepare features for ML models."""
        # Encode categorical variables
        if is_training:
            df['driver_encoded'] = self.driver_encoder.fit_transform(df['driver'])
            df['team_encoded'] = self.team_encoder.fit_transform(df['team'])
            df['track_encoded'] = self.track_encoder.fit_transform(df['track'])
        else:
            # Handle unseen categories during prediction
            df['driver_encoded'] = self.safe_transform(self.driver_encoder, df['driver'])
            df['team_encoded'] = self.safe_transform(self.team_encoder, df['team'])
            df['track_encoded'] = self.safe_transform(self.track_encoder, df['track'])
        
        # Select features
        feature_columns = [
            'driver_encoded', 'team_encoded', 'track_encoded', 'year',
            'quali_position', 'grid_position', 'driver_points_before', 
            'constructor_points_before', 'historical_pos_change',
            'track_pole_rate', 'track_overtake_difficulty', 'track_difficulty'
        ]
        
        # Handle race pace (might be 0 for prediction)
        if 'race_pace' in df.columns:
            df['race_pace_normalized'] = df['race_pace'] / df['race_pace'].replace(0, np.nan).mean()
            df['race_pace_normalized'] = df['race_pace_normalized'].fillna(1.0)
            feature_columns.append('race_pace_normalized')
        
        self.feature_names = feature_columns
        features = df[feature_columns].fillna(0)
        
        if is_training:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
            
        return features_scaled
    
    def safe_transform(self, encoder, data):
        """Safely transform data, handling unseen categories."""
        result = []
        for item in data:
            try:
                result.append(encoder.transform([item])[0])
            except ValueError:  # Unseen category
                result.append(0)  # Default to first category
        return result
    
    def train_models(self, df: pd.DataFrame):
        """Train multiple ML models and select the best one."""
        print("\nPreparing features for training...")
        X = self.prepare_features(df, is_training=True)
        y = df['target_position'].values
        
        print(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        best_score = float('inf')
        best_model_name = None
        
        print("\nTraining models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            
            print(f"    MAE: {mae:.3f}, RMSE: {rmse:.3f}, CV MAE: {cv_mae:.3f}")
            
            if mae < best_score:
                best_score = mae
                best_model_name = name
                self.best_model = model
        
        print(f"\nBest model: {best_model_name} (MAE: {best_score:.3f})")
        
        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
    
    def predict_race(self, year: int, track: str) -> pd.DataFrame:
        """Make predictions for a specific race - following your prediction pattern."""
        print(f"\nMaking predictions for {year} {track}...")
        
        # Use track mapping like in data collection
        track_mapping = {
            'Baku': ['Azerbaijan Grand Prix', 'Baku'],
            'Singapore': ['Singapore Grand Prix', 'Singapore'],
            'Austin': ['United States Grand Prix', 'Austin'],
            'Mexico City': ['Mexico City Grand Prix', 'Mexico City'],
            'São Paulo': ['Brazilian Grand Prix', 'São Paulo'],
            'Las Vegas': ['Las Vegas Grand Prix', 'Las Vegas'],
            'Qatar': ['Qatar Grand Prix', 'Qatar'],
            'Abu Dhabi': ['Abu Dhabi Grand Prix', 'Abu Dhabi']
        }
        
        track_names_to_try = track_mapping.get(track, [track])
        
        for track_name in track_names_to_try:
            try:
                # Load current qualifying data - following your pattern
                quali_session = ff1.get_session(year, track_name, 'Q')
                quali_session.load(laps=False, telemetry=False, weather=False, messages=False)
                quali_results = quali_session.results
                print(f"Successfully loaded qualifying data for {year} {track_name}")
                break
                
            except Exception as e:
                print(f"Could not load {track_name}: {e}")
                continue
        else:
            print(f"Could not load qualifying data for any variant of {track}")
            return pd.DataFrame()
        
        # Get current season data - following your season analysis pattern
        season_standings = self.get_season_standings_before_race(year, track_name)
        historical_performance = self.get_historical_performance(year, track)
        
        # Prepare prediction data
        prediction_data = []
        track_info = TRACK_DATA.get(track, {'pole_rate': 0.4, 'overtake_difficulty': 0.5})
        
        for _, row in quali_results.iterrows():
            driver = row['Abbreviation']
            team = row['TeamName']
            grid_pos = row['GridPosition'] if pd.notna(row['GridPosition']) else row['Position']
            
            features = {
                'driver': driver,
                'team': team,
                'track': track,  # Use standardized track name
                'year': year,
                'quali_position': row['Position'],
                'grid_position': grid_pos,
                'driver_points_before': season_standings['driver_points'].get(driver, 0),
                'constructor_points_before': season_standings['constructor_points'].get(team, 0),
                'historical_pos_change': historical_performance.get(driver, 0.0),
                'track_pole_rate': track_info['pole_rate'],
                'track_overtake_difficulty': track_info['overtake_difficulty'],
                'track_difficulty': (track_info['pole_rate'] + track_info['overtake_difficulty']) / 2,
                'race_pace': 0.0  # Unknown for prediction
            }
            
            prediction_data.append(features)
        
        pred_df = pd.DataFrame(prediction_data)
        X_pred = self.prepare_features(pred_df, is_training=False)
        
        # Make predictions
        predictions = self.best_model.predict(X_pred)
        
        # Create results dataframe
        results_df = pred_df.copy()
        results_df['predicted_position'] = np.round(predictions).astype(int)
        results_df['predicted_position'] = np.clip(results_df['predicted_position'], 1, 20)
        
        # Sort by predicted position
        results_df = results_df.sort_values('predicted_position').reset_index(drop=True)
        
        return results_df[['driver', 'team', 'quali_position', 'predicted_position', 
                         'driver_points_before', 'historical_pos_change']]


def main():
    """Main execution function."""
    # Configuration
    CURRENT_YEAR = 2025
    TRACK = 'Baku'
    TRAINING_START_YEAR = 2022
    TRAINING_END_YEAR = 2024
    
    # Tracks to include in training (must have track data)
    TRAINING_TRACKS = list(TRACK_DATA.keys())
    
    print("=== F1 Machine Learning Race Predictor ===")
    
    # Initialize predictor
    predictor = F1MLPredictor()
    
    # Enable F1 cache
    ff1.Cache.enable_cache('f1_cache')
    
    try:
        # Collect and prepare training data
        training_data = predictor.get_historical_data(
            TRAINING_START_YEAR, TRAINING_END_YEAR, TRAINING_TRACKS
        )
        
        if training_data.empty:
            print("No training data collected. Exiting.")
            return
        
        # Train models
        predictor.train_models(training_data)
        
        # Make predictions for current race
        predictions = predictor.predict_race(CURRENT_YEAR, TRACK)
        
        if not predictions.empty:
            print(f"\n=== ML PREDICTIONS FOR {CURRENT_YEAR} {TRACK} ===")
            print(predictions.to_string(index=False))
            
            # Create a clean final positions table (1-20)
            print(f"\n=== FINAL PREDICTED RACE ORDER (1-20) ===")
            print(f"{'Pos':<3} {'Driver':<4} {'Team':<20} {'Start':<5}")
            print("-" * 40)
            
            # Add actual final finishing position to the dataframe
            predictions['final_position'] = range(1, len(predictions) + 1)
            
            for idx, row in predictions.iterrows():
                final_pos = idx + 1
                driver = row['driver']
                team = row['team'][:20]  # Truncate team name if too long
                start_pos = int(row['quali_position'])
                print(f"{final_pos:<3} {driver:<4} {team:<20} P{start_pos}")
            
            print(f"\n=== PREDICTION SUMMARY ===")
            print(f"Predicted Winner: {predictions.iloc[0]['driver']} (started P{int(predictions.iloc[0]['quali_position'])})")
            print(f"Predicted Podium: {', '.join(predictions.head(3)['driver'].tolist())}")
            
            # Calculate position changes using ACTUAL final positions (positive = gained positions, negative = lost positions)
            predictions['position_change'] = predictions['quali_position'] - predictions['final_position']
            
            # Filter out drivers who didn't move significantly and get actual gainers/losers
            gainers = predictions[predictions['position_change'] > 0].nlargest(3, 'position_change')
            losers = predictions[predictions['position_change'] < 0].nsmallest(3, 'position_change')
            
            if not gainers.empty:
                print(f"\nBiggest Predicted Gainers:")
                for _, row in gainers.iterrows():
                    print(f"  {row['driver']}: P{int(row['quali_position'])} → P{row['final_position']} ({row['position_change']:+.0f})")
            
            if not losers.empty:
                print(f"\nBiggest Predicted Losers:")
                for _, row in losers.iterrows():
                    print(f"  {row['driver']}: P{int(row['quali_position'])} → P{row['final_position']} ({row['position_change']:+.0f})")
            
            # Show points implications
            top_points_scorers = predictions.head(10)
            print(f"\nPredicted Points Scorers (Top 10):")
            points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
            for idx, row in top_points_scorers.iterrows():
                points = points_system[idx]
                print(f"  P{idx+1}: {row['driver']} (+{points} points)")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
