import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
import google.generativeai as genai
from fpdf import FPDF
import os
import unicodedata

# 🔐 Setup Gemini API
genai.configure(api_key="AIzaSyAujZALUmAIgpUHjnVZ018dWw0oh8t55xo")  # Replace with your actual Gemini API key

# 🗞 BMI calculator
def calculate_bmi(weight, height):
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)

# 📊 Calorie plot
def generate_calorie_plot(target_calories):
    days = list(range(1, 8))
    burn = [target_calories] * 7
    plt.figure(figsize=(6, 3))
    plt.plot(days, burn, marker='o', color='orange')
    plt.title('Daily Calorie Burn Plan')
    plt.xlabel('Day')
    plt.ylabel('Calories to Burn')
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

# 🧠 Gemini LLM call
def generate_fitness_plan(user_input):
    prompt = f"""
You are a fitness and nutrition expert AI.

User details:
- Name: {user_input['name']}
- Age: {user_input['age']}
- Gender: {user_input['gender']}
- Height: {user_input['height']} cm
- Weight: {user_input['weight']} kg
- Goal: {user_input['goal']} {user_input['amount']} kg
- Diet Type: {user_input['diet_type']}
- Health Conditions: {user_input['diseases']}

Please respond with:
1. Greet the user using their name.
2. Show daily calories to burn to reach the goal.
3. Show BMI status (underweight/normal/overweight/obese).
4. Provide a 7-day {user_input['diet_type']} meal plan. Each day’s meal plan should include:
   - Day title
   - Breakfast (time and content)
   - Lunch (time and content)
   - Dinner (time and content)
5. Provide a 7-day exercise plan:
   - Each day's workout with type, purpose, time, sets/reps
   - Include image descriptions
6. Suggest foods and habits to avoid.
7. Precautions and exercises to avoid based on diseases.
Wrap each major section (greeting, BMI, calories, Day 1, Day 2, etc.) using clear subheadings (###) so they can be styled better.
"""
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model.generate_content(prompt)
    return response.text

# 🔤 Clean text to support Latin-1 encoding for PDF
def clean_text_for_pdf(text):
    text = unicodedata.normalize('NFKD', text)
    return text.encode('latin-1', 'ignore').decode('latin-1')

# 📝 Save text as PDF
def save_plan_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    cleaned_text = clean_text_for_pdf(text)
    for line in cleaned_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    temp_path = "temp_fitness_plan.pdf"
    pdf.output(temp_path)
    with open(temp_path, "rb") as f:
        pdf_data = f.read()
    os.remove(temp_path)
    return pdf_data

# 🌐 Streamlit UI
st.set_page_config(page_title="Gemini Fitness Planner", layout="wide")
st.title("\U0001F3CB\u200D\u2640\ufe0f Gemini AI Fitness & Diet Planner")

with st.form("fitness_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("\U0001F464 Name", placeholder="Enter your name")
        age = st.number_input("\U0001F382 Age", min_value=10, max_value=100, value=None, placeholder="Enter your age")
        gender = st.selectbox("\u26A7 Gender", ["Select", "Female", "Male", "Other"])
        diet_type = st.radio("\U0001F957 Diet Type", ["vegetarian", "non-vegetarian"], index=None)
    with col2:
        height = st.number_input("\U0001F4CF Height (cm)", value=None, placeholder="e.g. 160")
        weight = st.number_input("\u2696\ufe0f Weight (kg)", value=None, placeholder="e.g. 70")
        goal = st.selectbox("\U0001F3AF Goal", ["Select", "weight loss", "weight gain", "maintain weight"])
        amount = st.number_input("\U0001F4C9/\U0001F4C8 Target kg to lose/gain", value=None, placeholder="e.g. 5")

    diseases = st.text_area("\U0001F4CA Any health conditions (e.g., PCOS, Diabetes)?", placeholder="Type here")

    submitted = st.form_submit_button("\U0001F9E0 Generate My Plan")

if submitted:
    if "Select" in [gender, goal] or not all([name, age, height, weight, diet_type]):
        st.error("❗ Please fill in all required fields.")
    else:
        user_input = {
            "name": name,
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "goal": goal,
            "amount": amount,
            "diet_type": diet_type,
            "diseases": diseases
        }

        bmi = calculate_bmi(weight, height)
        st.markdown(f"<div style='background:#f5f5f5;border-radius:8px;padding:12px 18px;margin-top:10px;color:blue;margin-bottom:10px;'><b>✅ Your BMI:</b> {bmi}</div>", unsafe_allow_html=True)

        plot_img = generate_calorie_plot(500)
        st.image(plot_img, caption="🔥 Daily Calorie Burn (Estimated)")

        with st.spinner("⏳ Generating your personalized plan..."):
            result = generate_fitness_plan(user_input)

        st.markdown("---")
        st.markdown("### 📋 Your Personalized Fitness Plan")

        for section in result.split("###"):
            section = section.strip()
            if section:
                with st.expander(section.split("\n")[0]):
                    st.markdown("\n".join(section.split("\n")[1:]))

        pdf_buffer = save_plan_to_pdf(result)
        st.download_button("\U0001F4C4 Download Plan as PDF", data=pdf_buffer, file_name="fitness_plan.pdf", mime="application/pdf")

