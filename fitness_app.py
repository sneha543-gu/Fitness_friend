import streamlit as st
from llm_model import generate_fitness_plan, calculate_bmi, generate_calorie_plot, save_plan_to_pdf
from auth import register_user, authenticate_user
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import math
import joblib

# Load Random Forest model
model = joblib.load("random_forest_fitness_model.pkl")

# Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# Sidebar Navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to", ["Login / Signup", "Fitness Plan Generator", "Live Exercise Counter"])

# LOGIN / SIGNUP PAGE
if page == "Login / Signup":
    st.set_page_config(page_title="Login System")
    st.title("🔐 Welcome to Fitness App")

    if st.session_state.logged_in:
        st.success(f"✅ You are logged in as {st.session_state.user}")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user = ""
            st.rerun()
    else:
        choice = st.radio("Select Option", ["Login", "Sign Up"])
        if choice == "Sign Up":
            st.subheader("📝 Create Account")
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")
            if st.button("Register"):
                if register_user(new_user, new_pass):
                    st.success("Account created! Please login.")
                else:
                    st.error("Username already exists.")
        elif choice == "Login":
            st.subheader("🔐 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if authenticate_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.success(f"Welcome {username}")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

# FITNESS PLAN PAGE
elif page == "Fitness Plan Generator":
    if not st.session_state.logged_in:
        st.warning("Please login first.")
        st.stop()

    st.title("🏋️ Personalized Fitness Plan Generator")
    with st.form("fitness_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name",placeholder="Enter your name")
            age = st.number_input("Age",placeholder="Enter your age")
            height = st.number_input("Height (cm)",placeholder="Enter your height")
            weight = st.number_input("Weight (kg)",placeholder="Enter your Weight")
            target_weight = st.number_input("Target Weight (kg)",placeholder="Enter target Weight")
        with col2:
            goal = st.selectbox("Goal", ["Slect","weight loss", "weight gain", "maintain weight"])
            goal_change = abs(weight - target_weight)
            diet_type = st.radio("Diet Type", ["vegetarian", "non-vegetarian"],index=None)
            exercise_level = st.selectbox("Exercise Level", ["Select Level","Fast", "Medium", "Slow"])
            disease = st.selectbox("Any Disease?", ["None", "Diabetes", "BP", "Thyroid"])
            gender = st.selectbox("Gender", ["Select","Female", "Male", "Other"])

        submitted = st.form_submit_button("🧠 Generate My Plan")

    if submitted:
        # Predict days with Random Forest
        features = [
            age, height, weight, goal_change, target_weight,
            1 if goal == "weight gain" else 0,
            1 if goal == "weight loss" else 0,
            1 if goal == "maintain weight" else 0,
            1 if diet_type == "non-vegetarian" else 0,
            1 if diet_type == "vegetarian" else 0,
            1 if exercise_level == "Fast" else 0,
            1 if exercise_level == "Medium" else 0,
            1 if exercise_level == "Slow" else 0,
            1 if disease == "BP" else 0,
            1 if disease == "Diabetes" else 0,
            1 if disease == "Thyroid" else 0,
            1 if gender == "Female" else 0,
            1 if gender == "Male" else 0,
            1 if gender == "Other" else 0,
        ]
        days = int(model.predict([features])[0])

        user_input = {
            "name": name,
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "goal": goal,
            "amount": goal_change,
            "diet_type": diet_type,
            "diseases": disease
        }

        bmi = calculate_bmi(weight, height)
        st.success(f"✅ Hello {name}, based on your goal, you will reach your target in ~{days} days!")
        st.info(f"📊 Your BMI is: {bmi}")

        st.image(generate_calorie_plot(500), caption="🔥 Daily Calorie Burn Plan")

        with st.spinner("Generating LLM meal & exercise plan..."):
            plan = generate_fitness_plan(user_input)

        st.markdown("### 🧾 Personalized AI Plan")
        for section in plan.split("###"):
            section = section.strip()
            if section:
                with st.expander(section.split("\n")[0]):
                    st.markdown("\n".join(section.split("\n")[1:]))

        st.download_button("📥 Download Plan as PDF", save_plan_to_pdf(plan), "fitness_plan.pdf")

# SQUAT COUNTER PAGE
elif page == "Live Exercise Counter":
    st.set_page_config(page_title="Squat Counter")
    st.title("🤸 Real-Time Squat Counter")
    st.markdown("This uses webcam and pose detection to count squats in real-time.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    class SquatCounter(VideoTransformerBase):
        def __init__(self):
            self.squat_count = 0
            self.squat_position = None

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark

                try:
                    h, w, _ = img.shape
                    hip = [landmarks[24].x * w, landmarks[24].y * h]
                    knee = [landmarks[26].x * w, landmarks[26].y * h]
                    ankle = [landmarks[28].x * w, landmarks[28].y * h]

                    angle = math.degrees(
                        math.atan2(ankle[1] - knee[1], ankle[0] - knee[0]) -
                        math.atan2(hip[1] - knee[1], hip[0] - knee[0])
                    )
                    angle = abs(angle)

                    if angle < 90:
                        if self.squat_position != 'down':
                            self.squat_position = 'down'
                    elif angle > 160:
                        if self.squat_position == 'down':
                            self.squat_count += 1
                            self.squat_position = 'up'

                    cv2.putText(img, f'Angle: {int(angle)}', (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(img, f'Count: {self.squat_count}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                except:
                    pass

            return img

    webrtc_streamer(
        key="squat",
        video_processor_factory=SquatCounter,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
