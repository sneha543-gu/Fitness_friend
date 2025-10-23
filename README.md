# Fitness_friend
 

**Fitness Friend** is an AI-based fitness web app built with **Streamlit**.  
It predicts how many days you need to lose, gain, or maintain your weight using a **Random Forest model**, and provides a **personalized 7-day meal and exercise plan**.  
It also includes a **live exercise counter** using your webcam and pose detection.



##  Features
- Login and Signup system (SQL-based)
- Predict days to reach fitness goals (ML – Random Forest)
- BMI calculator and calorie chart
- AI-generated 7-day meal plan
- Personalized exercise plan
- Live squat counter using webcam (MediaPipe + OpenCV)
- Download plan as PDF



##  Technologies Used 

 Python : Core programming language used to build the entire application. 
 Streamlit : Creates an interactive and user-friendly web interface for the fitness app. 
 **Random Forest (Scikit-learn)** : Machine learning model used to predict the number of days needed to achieve the fitness goal. 
 **MediaPipe** : Performs real-time human pose detection for counting exercises. 
 **OpenCV** : Handles webcam input and video frame processing for live exercise tracking. 
 **CVZone** : Simplifies interaction between OpenCV and MediaPipe for accurate movement detection. 
 **Matplotlib** : Generates calorie burn charts and data visualizations. 
 **SQL Database** : Stores and manages user login/signup credentials securely. 
 **ReportLab** : Used to generate and download the personalized fitness plan as a PDF file. 



##  How to Run

To run this project on your system, follow these simple steps:

```bash
# 1. Clone this repository
git clone https://github.com/your-username/Fitness-Friend.git

# 2. Move into the project folder
cd Fitness-Friend

# 3. Install all required dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
