import streamlit as st
import json
import csv
import datetime
import random
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
import base64
import os
import pandas as pd

# Load the configurations
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# Save the configurations
def save_config(config, config_file):
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def load_csv_as_df(file_path):
    return pd.read_csv(file_path)

def display_historical_data(df):
    st.subheader("Historical Data (Sample)")
    st.dataframe(df.head(10))  # Display first 10 rows
    st.write(f"Total records: {len(df)}")

def display_competitor_data(df):
    st.subheader("Competitor Data (Sample)")
    st.dataframe(df.head(10))  # Display first 10 rows
    st.write(f"Total records: {len(df)}")

def display_event_data(df):
    st.subheader("Event Calendar (All Events)")
    st.dataframe(df)
    st.write(f"Total events: {len(df)}")

# Define the Dynamic Pricing Engine class
class AdvancedDynamicPricingEngine:
    def __init__(self, config, historical_data_file, competitor_data_file, event_calendar_file, strategy):
        self.config = config
        self.historical_data = self.load_historical_data(historical_data_file)
        self.competitor_data = self.load_competitor_data(competitor_data_file)
        self.event_calendar = self.load_event_calendar(event_calendar_file)
        self.pricing_model = self.train_pricing_model()
        self.strategy = strategy

    def load_historical_data(self, historical_data_file):
        data = defaultdict(lambda: defaultdict(dict))
        with open(historical_data_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
                room_type = row['room_type']
                data[date][room_type]['price'] = float(row['price'])
                data[date][room_type]['occupancy'] = float(row['occupancy'])
        return data

    def load_competitor_data(self, competitor_data_file):
        data = defaultdict(dict)
        with open(competitor_data_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
                for competitor in reader.fieldnames[1:]:
                    data[date][competitor] = float(row[competitor])
        return data

    def load_event_calendar(self, event_calendar_file):
        events = defaultdict(list)
        with open(event_calendar_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
                events[date].append(row['event'])
        return events

    def train_pricing_model(self):
        X = []
        y = []
        for date, room_data in self.historical_data.items():
            for room_type, data in room_data.items():
                features = [
                    date.weekday(),
                    self.get_season_multiplier(date),
                    data['occupancy'],
                    len(self.event_calendar.get(date, [])),
                    sum(self.competitor_data.get(date, {}).values()) / len(self.competitor_data.get(date, {})) if date in self.competitor_data else self.config['base_prices'][room_type],
                    self.config['room_types'].index(room_type)
                ]
                X.append(features)
                y.append(data['price'])

        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X, y)
        return model

    def get_season_multiplier(self, date):
        month = date.month
        for season, months in self.config['seasons'].items():
            if month in months:
                return self.config['season_multipliers'][season]
        return 1.0

    def calculate_static_price(self, date, room_type):
        base_price = self.config['base_prices'][room_type]
        season_multiplier = self.get_season_multiplier(date)
        weekend_multiplier = 1.2 if date.weekday() >= 5 else 1.0
        static_price = base_price * season_multiplier * weekend_multiplier
        return round(static_price, -2)  # Round to nearest 100 INR

    def calculate_dynamic_price(self, date, room_type, occupancy_rate):
        base_price = self.config['base_prices'][room_type]
        min_price = self.config['min_prices'][room_type]
        max_price = base_price * 2  # Allow for up to 100% increase

        features = [
            date.weekday(),
            self.get_season_multiplier(date),
            occupancy_rate,
            len(self.event_calendar.get(date, [])),
            sum(self.competitor_data.get(date, {}).values()) / len(self.competitor_data.get(date, {})) if date in self.competitor_data else base_price,
            self.config['room_types'].index(room_type)
        ]

        predicted_price = self.pricing_model.predict([features])[0]

        # Adjust price based on occupancy
        if occupancy_rate > 0.8:
            predicted_price *= 1.2
        elif occupancy_rate < 0.4:
            predicted_price *= 0.9

        # Adjust price based on competitor pricing
        avg_competitor_price = sum(self.competitor_data.get(date, {}).values()) / len(self.competitor_data.get(date, {})) if date in self.competitor_data else base_price
        if predicted_price < avg_competitor_price * 0.9:
            predicted_price = avg_competitor_price * 0.95
        elif predicted_price > avg_competitor_price * 1.2:
            predicted_price = avg_competitor_price * 1.15

        # Adjust for weekends and events
        if date.weekday() >= 5:  # Weekend
            predicted_price *= 1.1
        if self.event_calendar.get(date):
            predicted_price *= 1.2  # Increased multiplier for events

        # Adjust based on strategy
        if self.strategy == 'maximize_revenue':
            predicted_price *= 1.05  # Slight increase to maximize revenue
        elif self.strategy == 'maximize_occupancy':
            predicted_price *= 0.95  # Slight decrease to attract more bookings

        # Ensure price is within allowed range
        final_price = max(min(round(predicted_price, -2), max_price), min_price)

        return final_price, self.generate_price_explanation(date, room_type, occupancy_rate, final_price, avg_competitor_price)

    def generate_price_explanation(self, date, room_type, occupancy_rate, final_price, avg_competitor_price):
        base_price = self.config['base_prices'][room_type]
        static_price = self.calculate_static_price(date, room_type)

        explanation = f"Price for {room_type} on {date}: ₹{final_price}\n"
        explanation += f"Base price: ₹{base_price}\n"
        explanation += f"Static price: ₹{static_price}\n"
        explanation += f"Occupancy rate: {occupancy_rate:.2f}\n"
        explanation += f"Average competitor price: ₹{avg_competitor_price:.2f}\n"
        explanation += f"Pricing strategy: {self.strategy}\n"

        if occupancy_rate > 0.8:
            explanation += "High occupancy rate, increased price.\n"
        elif occupancy_rate < 0.4:
            explanation += "Low occupancy rate, decreased price.\n"

        if final_price > static_price:
            explanation += "Dynamic price is higher due to favorable conditions.\n"
        elif final_price < static_price:
            explanation += "Dynamic price is lower to stay competitive.\n"

        if date.weekday() >= 5:
            explanation += "Weekend pricing applied.\n"

        if self.event_calendar.get(date):
            explanation += f"Event pricing applied for: {', '.join(self.event_calendar.get(date))}\n"

        if self.strategy == 'maximize_revenue':
            explanation += "Price slightly increased to maximize revenue.\n"
        elif self.strategy == 'maximize_occupancy':
            explanation += "Price slightly decreased to maximize occupancy.\n"

        return explanation

    def run_simulation(self, start_date, days):
        results = []
        current_date = start_date
        for _ in range(days):
            daily_result = {'date': current_date, 'day_type': 'Weekend' if current_date.weekday() >= 5 else 'Weekday', 'events': self.event_calendar.get(current_date, [])}

            for room_type in self.config['room_types']:
                if current_date in self.historical_data and room_type in self.historical_data[current_date]:
                    occupancy = self.historical_data[current_date][room_type]['occupancy']
                else:
                    occupancy = random.uniform(0.5, 1.0)

                dynamic_price, price_explanation = self.calculate_dynamic_price(current_date, room_type, occupancy)
                static_price = self.calculate_static_price(current_date, room_type)

                daily_result[f'{room_type}_dynamic_price'] = dynamic_price
                daily_result[f'{room_type}_static_price'] = static_price
                daily_result[f'{room_type}_occupancy'] = occupancy
                daily_result[f'{room_type}_price_explanation'] = price_explanation

            if current_date in self.competitor_data:
                daily_result['competitor_avg'] = sum(self.competitor_data[current_date].values()) / len(self.competitor_data[current_date])
            else:
                daily_result['competitor_avg'] = random.uniform(7000, 9000)

            results.append(daily_result)
            current_date += datetime.timedelta(days=1)
        return results

    def calculate_metrics(self, simulation_results):
        dynamic_revenue = 0
        static_revenue = 0
        dynamic_occupancy = 0
        static_occupancy = 0
        total_rooms = 0

        for result in simulation_results:
            for room_type in self.config['room_types']:
                rooms = self.config['room_counts'][room_type]
                total_rooms += rooms
                occupancy = result[f'{room_type}_occupancy']
                dynamic_revenue += result[f'{room_type}_dynamic_price'] * occupancy * rooms
                static_revenue += result[f'{room_type}_static_price'] * occupancy * rooms
                dynamic_occupancy += occupancy * rooms
                static_occupancy += occupancy * rooms

        total_room_nights = total_rooms * len(simulation_results)

        dynamic_adr = dynamic_revenue / dynamic_occupancy
        static_adr = static_revenue / static_occupancy
        dynamic_revpar = dynamic_revenue / total_room_nights
        static_revpar = static_revenue / total_room_nights
        dynamic_occupancy_rate = dynamic_occupancy / total_room_nights
        static_occupancy_rate = static_occupancy / total_room_nights

        return {
            'dynamic': {
                'revenue': dynamic_revenue,
                'adr': dynamic_adr,
                'revpar': dynamic_revpar,
                'occupancy_rate': dynamic_occupancy_rate
            },
            'static': {
                'revenue': static_revenue,
                'adr': static_adr,
                'revpar': static_revpar,
                'occupancy_rate': static_occupancy_rate
            }
        }

    def generate_charts(self, simulation_results):
        dates = [result['date'] for result in simulation_results]
        dynamic_prices = {room_type: [result[f'{room_type}_dynamic_price'] for result in simulation_results] for room_type in self.config['room_types']}
        static_prices = {room_type: [result[f'{room_type}_static_price'] for result in simulation_results] for room_type in self.config['room_types']}
        competitor_avgs = [result['competitor_avg'] for result in simulation_results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15))

        # Price comparison chart
        for room_type in self.config['room_types']:
            ax1.plot(dates, dynamic_prices[room_type], label=f'{room_type} Dynamic')
            ax1.plot(dates, static_prices[room_type], label=f'{room_type} Static', linestyle='--')
        ax1.plot(dates, competitor_avgs, label='Avg Competitor', color='black', linewidth=2)
        ax1.set_title('Price Comparison by Room Type')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (INR)')
        ax1.legend()

        # Revenue comparison chart
        dynamic_revenue = [sum(result[f'{room_type}_dynamic_price'] * result[f'{room_type}_occupancy'] * self.config['room_counts'][room_type] for room_type in self.config['room_types']) for result in simulation_results]
        static_revenue = [sum(result[f'{room_type}_static_price'] * result[f'{room_type}_occupancy'] * self.config['room_counts'][room_type] for room_type in self.config['room_types']) for result in simulation_results]
        ax2.plot(dates, dynamic_revenue, label='Dynamic Revenue')
        ax2.plot(dates, static_revenue, label='Static Revenue')
        ax2.set_title('Daily Revenue Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Revenue (INR)')
        ax2.legend()

        # Save the chart to a base64 encoded string
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return chart_base64

# Function to generate or load historical data
def get_historical_data(config, start_date, days, force_regenerate=False):
    historical_data_file = "historical_data.csv"
    if not os.path.exists(historical_data_file) or force_regenerate:
        with open(historical_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'room_type', 'price', 'occupancy'])
            current_date = start_date - datetime.timedelta(days=days)
            for _ in range(days):
                for room_type in config['room_types']:
                    base_price = config['base_prices'][room_type]
                    price = base_price * random.uniform(0.8, 1.2)
                    occupancy = random.uniform(0.5, 1.0)
                    writer.writerow([current_date.strftime('%Y-%m-%d'), room_type, price, occupancy])
                current_date += datetime.timedelta(days=1)
    return historical_data_file

# Function to generate or load competitor data
def get_competitor_data(start_date, days, force_regenerate=False):
    competitor_data_file = "competitor_data.csv"
    if not os.path.exists(competitor_data_file) or force_regenerate:
        with open(competitor_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'competitor_A', 'competitor_B', 'competitor_C'])
            current_date = start_date - datetime.timedelta(days=days)
            for _ in range(days):
                row = [current_date.strftime('%Y-%m-%d')]
                row.extend([random.uniform(7000, 9000) for _ in range(3)])
                writer.writerow(row)
                current_date += datetime.timedelta(days=1)
    return competitor_data_file

# Function to generate or load event calendar
def get_event_calendar(start_date, days, force_regenerate=False):
    event_calendar_file = "event_calendar.csv"
    if not os.path.exists(event_calendar_file) or force_regenerate:
        with open(event_calendar_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'event'])
            current_date = start_date - datetime.timedelta(days=days)
            events = ['conference', 'festival', 'sports_event', 'holiday']
            for _ in range(days):
                if random.random() < 0.1 or current_date.strftime('%m-%d') in ['08-15', '08-19']:  # 10% chance of an event + specific dates
                    event = 'Independence Day' if current_date.strftime('%m-%d') == '08-15' else 'Raksha Bandhan' if current_date.strftime('%m-%d') == '08-19' else random.choice(events)
                    writer.writerow([current_date.strftime('%Y-%m-%d'), event])
                current_date += datetime.timedelta(days=1)
    return event_calendar_file

# Streamlit app begins here
def main():
    st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="🏨", layout="wide")
    
    st.title("🏨 Advanced Dynamic Pricing Engine")
    st.markdown("Welcome to the Advanced Dynamic Pricing Engine! This application demonstrates a dynamic pricing model for a hotel, where you can select different strategies and visualize their impact on revenue and pricing.")

    # Load or create config
    config_file = 'pricing_config.json'
    if os.path.exists(config_file):
        config = load_config(config_file)
    else:
        config = {
            "base_prices": {"Standard": 8000, "Deluxe": 10000, "Suite": 15000},
            "min_prices": {"Standard": 6000, "Deluxe": 7500, "Suite": 11000},
            "room_types": ["Standard", "Deluxe", "Suite"],
            "room_counts": {"Standard": 150, "Deluxe": 100, "Suite": 50},
            "seasons": {
                "high": [10, 11, 12, 1, 2],
                "low": [5, 6, 7, 8],
                "shoulder": [3, 4, 9]
            },
            "season_multipliers": {
                "high": 1.3,
                "low": 0.8,
                "shoulder": 1.0
            }
        }
        save_config(config, config_file)

    with st.sidebar:
        st.header("Simulation Settings")
        strategy = st.selectbox("Choose a Pricing Strategy", ["maximize_revenue", "maximize_occupancy"], help="Select the strategy that best aligns with your goals.")
        start_date = st.date_input("Select Start Date", datetime.date(2024, 8, 14))
        days = st.slider("Number of Days to Simulate", min_value=7, max_value=365, value=30)

        st.header("Hotel Configuration")
        for room_type in config['room_types']:
            config['base_prices'][room_type] = st.number_input(f"Base Price for {room_type}", value=config['base_prices'][room_type], step=100)
            config['min_prices'][room_type] = st.number_input(f"Minimum Price for {room_type}", value=config['min_prices'][room_type], step=100)
            config['room_counts'][room_type] = st.number_input(f"Number of {room_type} Rooms", value=config['room_counts'][room_type], step=1)

        st.header("Season Configuration")
        for season in config['seasons']:
            config['season_multipliers'][season] = st.slider(f"{season.capitalize()} Season Multiplier", min_value=0.5, max_value=2.0, value=config['season_multipliers'][season], step=0.1)

        if st.button("Save Configuration"):
            save_config(config, config_file)
            st.success("Configuration saved successfully!")

    # Generate or load data
    historical_data_file = get_historical_data(config, start_date, 365)
    competitor_data_file = get_competitor_data(start_date, 365)
    event_calendar_file = get_event_calendar(start_date, 365)

    # Display the generated data
    st.header("Generated Data for Simulation")
    
    historical_df = load_csv_as_df(historical_data_file)
    display_historical_data(historical_df)
    
    competitor_df = load_csv_as_df(competitor_data_file)
    display_competitor_data(competitor_df)
    
    event_df = load_csv_as_df(event_calendar_file)
    display_event_data(event_df)

    if st.button("Run Simulation"):
        # Initialize and run the pricing engine
        engine = AdvancedDynamicPricingEngine(config, historical_data_file, competitor_data_file, event_calendar_file, strategy)
        simulation_results = engine.run_simulation(start_date, days)
        metrics = engine.calculate_metrics(simulation_results)
        chart_base64 = engine.generate_charts(simulation_results)

        # Display results
        st.subheader("📅 Simulation Results (First 7 Days)")
        headers = ["Date", "Day Type", "Events"] + [f"{room_type} Dynamic" for room_type in config['room_types']] + [f"{room_type} Static" for room_type in config['room_types']] + ["Avg Competitor"]
        table_data = [headers] + [
            [
                result['date'].strftime('%Y-%m-%d'),
                result['day_type'],
                ', '.join(result['events']) if result['events'] else 'None'
            ] +
            [f"₹{result[f'{room_type}_dynamic_price']:,}" for room_type in config['room_types']] +
            [f"₹{result[f'{room_type}_static_price']:,}" for room_type in config['room_types']] +
            [f"₹{result['competitor_avg']:,.2f}"]
            for result in simulation_results[:7]
        ]
        st.table(table_data)

        st.subheader("📊 Key Metrics Comparison")
        metrics_table = [
            ["Metric", "Dynamic Pricing", "Static Pricing", "Improvement"],
            ["Total Revenue", f"₹{metrics['dynamic']['revenue']:,.2f}", f"₹{metrics['static']['revenue']:,.2f}",
             f"{(metrics['dynamic']['revenue'] - metrics['static']['revenue']) / metrics['static']['revenue'] * 100:.2f}%"],
            ["ADR", f"₹{metrics['dynamic']['adr']:,.2f}", f"₹{metrics['static']['adr']:,.2f}",
             f"{(metrics['dynamic']['adr'] - metrics['static']['adr']) / metrics['static']['adr'] * 100:.2f}%"],
            ["RevPAR", f"₹{metrics['dynamic']['revpar']:,.2f}", f"₹{metrics['static']['revpar']:,.2f}",
             f"{(metrics['dynamic']['revpar'] - metrics['static']['revpar']) / metrics['static']['revpar'] * 100:.2f}%"],
            ["Occupancy Rate", f"{metrics['dynamic']['occupancy_rate']:.2%}", f"{metrics['static']['occupancy_rate']:.2%}",
             f"{(metrics['dynamic']['occupancy_rate'] - metrics['static']['occupancy_rate']) / metrics['static']['occupancy_rate'] * 100:.2f}%"]
        ]
        st.table(metrics_table)

        st.subheader("📈 Revenue and Pricing Charts")
        st.image(f"data:image/png;base64,{chart_base64}")

if __name__ == "__main__":
    main()