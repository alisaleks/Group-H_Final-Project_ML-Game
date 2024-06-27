import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config for better UI
st.set_page_config(page_title="FarmAI - Predictive Farming Game", page_icon="🌾", layout="wide")

# Sample data for demonstration
def get_sample_data(num_samples=100):
    np.random.seed(42)
    soil_moisture = np.random.uniform(1, 100, num_samples)
    temperature = np.random.uniform(10, 51, num_samples)
    humidity = np.random.uniform(0, 100, num_samples)
    sunlight_hours = np.random.choice([8, 5, 4, 3], num_samples)  # full sun, part sun, part shade, full shade
    
    # Ideal conditions
    ideal_soil_moisture = 64
    ideal_temperature = 22
    ideal_humidity = 50
    
    # Parabolic yield function
    crop_yield = (
        -0.2 * (soil_moisture - ideal_soil_moisture) ** 2 +
        -0.5 * (temperature - ideal_temperature) ** 2 +
        -0.2 * (humidity - ideal_humidity) ** 2 +
        -35.0 * (sunlight_hours - 8) ** 2 +
        1000 + np.random.normal(0, 5, num_samples)  # Adding a base value for yield and noise
    )
    
    data = pd.DataFrame({
        'Soil Moisture': soil_moisture,
        'Temperature': temperature,
        'Humidity': humidity,
        'Sunlight Hours': sunlight_hours,
        'Crop Yield': crop_yield
    })
    return data

# Prediction function
def predict_yield(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return predictions, mse, r2

# Timer function
def update_timer():
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    elapsed_time = time.time() - st.session_state.start_time
    remaining_time = st.session_state.time_limit - elapsed_time
    if remaining_time <= 0:
        remaining_time = 0
        st.session_state.page = "no_attempts_left"
    return remaining_time

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"
    st.session_state.attempts_left = 2  # Initialize attempts
    st.session_state.submissions = []
    st.session_state.input_values = []
    st.session_state.total_cost = 0
    st.session_state.total_yield = 0
    st.session_state.show_feedback = False  # Initialize show_feedback
    st.session_state.time_limit = 60  # Default time limit in seconds
    st.session_state.show_tips = False  # Initialize show_tips

def initialize_data():
    st.session_state.data = get_sample_data()
    st.session_state.X = st.session_state.data.drop(columns=["Crop Yield"])
    st.session_state.y = st.session_state.data["Crop Yield"]
    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
        st.session_state.X, st.session_state.y, test_size=0.2, random_state=42
    )

# Sidebar
with st.sidebar:
    st.image("https://img.freepik.com/premium-photo/farmer-boy-3d-cartoon-character-white-background_82802-10790.jpg", use_column_width=True)
    st.markdown("### Navigate through the game")
    if st.button("Start Over"):
        st.session_state.page = "landing"
        st.session_state.attempts_left = 2  # Reset attempts
        st.session_state.start_time = None  # Reset start time
        initialize_data()  # Reinitialize data

# Main App
st.title("Welcome to FarmAI - Predictive Farming Game")
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

# Act 1: Seeds of Change
if st.session_state.page == "landing":
    st.subheader("Act 1: Seeds of Change")
    st.markdown("<p class='big-font'>Meet Carlos, a young farmer in the outskirts of Barcelona, who inherits his family's struggling farm. Determined to revitalize the land using modern methods, Carlos dives into the world of predictive farming.</p>", unsafe_allow_html=True)
    st.write("Optimize Your Crop Yield Using AI and Predictive Modeling")
    attempts = st.number_input("Set the number of attempts", min_value=1, max_value=10, value=st.session_state.attempts_left)
    time_limit = st.number_input("Set the time limit (in seconds)", min_value=10, max_value=3600, value=60)
    st.session_state.attempts = attempts
    st.session_state.time_limit = time_limit
    
    if st.button("Start Carlos's Journey"):
        st.session_state.page = "play"
        st.session_state.attempts_left = attempts  # Set the attempts left
        initialize_data()  # Initialize data
        st.session_state.start_time = time.time()  # Record the start time
    if st.button("More Info"):
        st.session_state.page = "info"

# Info Page
if st.session_state.page == "info":
    st.subheader("About the Game")
    st.write("""
    "FarmAI - Predictive Farming Game" is an educational game designed to help users understand the concept of regression 
    and predictive modeling in machine learning. The game places the user in the role of Carlos, a farm manager who 
    uses AI tools to predict and optimize crop yields based on various environmental factors. 
    """)
    if st.button("Back"):
        st.session_state.page = "landing"

# Act 2: Nurturing Growth
if st.session_state.page == "play":
    remaining_time = update_timer()
    st.subheader("Act 2: Nurturing Growth")
    st.progress(remaining_time / st.session_state.time_limit)
    st.metric("Time Left", f"{int(remaining_time)} seconds")
    st.metric("Attempts Left", st.session_state.attempts_left)
    st.write("Carlos faces challenges like fluctuating soil moisture, unpredictable weather, and varying sunlight. Help Carlos make strategic decisions to improve the farm's yield.")
    
    if st.session_state.page == "no_attempts_left":
        st.write("Time's up!")
    else:
        X_train = st.session_state.X_train
        data = st.session_state.data
        trained_model = RandomForestRegressor()  
        trained_model.fit(X_train, st.session_state.y_train)
        
        # Create sliders for each factor
        optimal_values = {}
        total_cost = 0
        cost_budget = 2500  # Arbitrary budget for the game
        for feature in X_train.columns:
            slider_value = st.slider(f"Select optimal value for {feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
            optimal_values[feature] = slider_value
            total_cost += slider_value * 10  # Simplified cost calculation

        total_cost = round(total_cost, 2)
        st.write("Total cost of your selections:", total_cost)
        st.write("Cost budget:", cost_budget)

        submit_button = st.button("Submit")

        if submit_button:
            if total_cost > cost_budget:
                st.error("You have exceeded the cost budget! Adjust your values to stay within the budget.")
            else:
                # Predict crop yield based on player's selections
                optimal_values_df = pd.DataFrame([optimal_values])[X_train.columns]  # Ensure correct column order
                predicted_yield = trained_model.predict(optimal_values_df)
                predicted_yield = round(predicted_yield[0], 2)
                st.session_state.submissions.append(predicted_yield)
                st.session_state.input_values.append(optimal_values)
                st.session_state.total_cost += total_cost
                st.session_state.total_yield += predicted_yield

                st.success(f"Predicted crop yield based on your selections: {predicted_yield}")
                st.session_state.start_time = time.time()  # Reset the timer for the next attempt
                st.session_state.page = "model_selection"

# Act 3: Harvesting Insights
if st.session_state.page == "model_selection":
    remaining_time = update_timer()
    st.subheader("Act 3: Harvesting Insights")
    st.progress(remaining_time / st.session_state.time_limit)
    st.metric("Time Left", f"{int(remaining_time)} seconds")
    st.metric("Attempts Left", st.session_state.attempts_left)
    st.write("Carlos learns valuable lessons about model suitability and data interpretation. Help him choose the best regression models.")
    
    if st.session_state.page == "no_attempts_left":
        st.write("Time's up!")
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }
        
        selected_models = st.multiselect("Select Regression Models", list(models.keys()), default=list(models.keys()))
        
        model_info = {
            "Linear Regression": " A linear approach to modeling the relationship between a dependent variable and one or more independent variables.",
            "Decision Tree": " A non-linear model that splits data into subsets based on feature values, forming a tree-like structure.",
            "Random Forest": " An ensemble method using multiple decision trees to improve predictive performance and reduce overfitting."
        }

        for model_name in selected_models:
            st.info(f"**{model_name}**: {model_info[model_name]}")
        
        if st.button("Predict"):
            st.session_state.page = "prediction"
            st.session_state.selected_models = selected_models
            st.session_state.models = models  # Store models in session state

# Prediction Page
if st.session_state.page == "prediction":
    remaining_time = update_timer()
    st.subheader("Prediction Results")
    st.progress(remaining_time / st.session_state.time_limit)
    st.metric("Time Left", f"{int(remaining_time)} seconds")
    st.metric("Attempts Left", st.session_state.attempts_left)
    
    if st.session_state.page == "no_attempts_left":
        st.write("Time's up!")
    else:
        models = st.session_state.models
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        
        results = []
        
        for selected_model in st.session_state.selected_models:
            model = models[selected_model]
            predictions, mse, r2 = predict_yield(model, X_train, X_test, y_train, y_test)
            
            results.append({
                "Model": selected_model,
                "Mean Squared Error": round(mse, 2),
                "R2 Score": round(r2, 2),
            })
        
        results_df = pd.DataFrame(results)
        st.table(results_df)
        
        # Add a button to show tips for metrics
        if st.button("Show Tips"):
            st.session_state.show_tips = not st.session_state.show_tips
        
        if st.session_state.show_tips:
            tips = """
            <div>
                <b>Tips:</b>
                <ul>
                    <li><b>Mean Squared Error (MSE):</b> Measures the average of the squares of the errors. Lower values are better.</li>
                    <li><b>R2 Score:</b> Represents the proportion of variance explained by the model. Higher values are better.</li>
                </ul>
            </div>
            """
            st.markdown(tips, unsafe_allow_html=True)
        
        # Determining the best model based on R2 Score, falling back to MSE if needed
        best_model_info = max(results, key=lambda x: (x['R2 Score'], -x['Mean Squared Error']))
        best_model = best_model_info['Model']

        selected_model = st.radio("Choose the best model:", [result["Model"] for result in results])
        
        if st.button("Choose this model"):
            if selected_model == best_model:
                st.session_state.page = "results"
                st.session_state.best_model = selected_model
                st.session_state.results_df = results_df
                st.session_state.trained_model = models[selected_model]
                st.session_state.attempts_left -= 1
                st.success(f"Congratulations! You chose the best model based on the metrics: {selected_model}.")
            else:
                st.session_state.attempts_left -= 1
                reason = "Decision Tree models often overfit the training data, capturing noise and details that do not generalize well to unseen data.  This can lead to high variance and poor performance on new data." if selected_model == "Decision Tree" else "the selected model is not the best choice based on the provided metrics."

                st.error(f"It was a bad decision because {reason}")
                if st.session_state.attempts_left > 0:
                    st.session_state.show_feedback = True  # Show feedback on why the choice was poor
                    st.session_state.reason = reason
                    st.session_state.start_time = time.time()  # Reset the timer for the next attempt
                    st.session_state.page = "model_selection_again"
                else:
                    st.session_state.page = "no_attempts_left"

# Display feedback after wrong choice
if st.session_state.page == "model_selection_again":
    remaining_time = update_timer()
    st.write(f"Time left: {int(remaining_time)} seconds")
    st.write(f"Attempts left: {st.session_state.attempts_left}")
    
    if st.session_state.page == "no_attempts_left":
        st.write("Time's up!")
    else:
        if st.session_state.show_feedback:
            st.write(f"Try again. Attempts left: {st.session_state.attempts_left}")
            st.session_state.show_feedback = False
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }
        selected_models = st.multiselect("Select Regression Models", list(models.keys()), default=list(models.keys()))
        if st.button("Predict Again"):
            st.session_state.selected_models = selected_models
            st.session_state.page = "prediction"

# No Attempts Left Page
if st.session_state.page == "no_attempts_left":
    remaining_time = update_timer()  # Get the updated remaining time even in this page
    st.error(f"You have no attempts left or time's up. Game over. Attempts left: {st.session_state.attempts_left}")
    if st.button("Play Again"):
        st.session_state.page = "landing"
        st.session_state.attempts_left = 2  # Reset attempts
        st.session_state.start_time = None  # Reset start time
        initialize_data()  # Reinitialize data

# Act 4: Thriving Fields
if st.session_state.page == "results":
    st.subheader("Act 4: Thriving Fields")
    st.balloons()
    st.write("Carlos reaps the rewards of his hard work and strategic decisions, achieving a record yield.")
    st.write(f"Congratulations! You successfully selected the best model based on performance metrics: {st.session_state.best_model}.")
    st.table(st.session_state.results_df)

    # Visualizations
    model = st.session_state.trained_model
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    if isinstance(model, (DecisionTreeRegressor, RandomForestRegressor)):
        feature_importances = model.feature_importances_
        features = X_train.columns

        fig, ax = plt.subplots()
        ax.barh(features, feature_importances)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        st.pyplot(fig)
    elif isinstance(model, LinearRegression):
        # Generate bar plot for top 3 most important features
        coefs = pd.Series(model.coef_, index=X_train.columns).sort_values(key=abs, ascending=False)
        top_3_features = coefs.head(3)

        fig, ax = plt.subplots()
        top_3_features.plot(kind='barh', ax=ax)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Top 3 Most Important Features')
        st.pyplot(fig)
    else:
        st.write("The selected model does not support feature importance visualization.")
    
    st.write("""
    ### Conclusion: A Sustainable Future
    With newfound knowledge, Carlos becomes a pioneer in predictive farming, inspiring others in Barcelona. His success story spreads, promoting sustainable agriculture and innovation in the community.

    Join Carlos in 'FarmAI' — where each decision shapes a prosperous future.
    """)
    if st.button("Play Again"):
        st.session_state.page = "landing"
        st.session_state.attempts_left = 2  # Reset attempts
        st.session_state.start_time = None  # Reset start time
        initialize_data()  # Reinitialize data
