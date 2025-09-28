# 🎨 AI Artist Web App

A simple yet powerful web application that transforms your photos into artistic masterpieces using **Neural Style Transfer (NST)**.  

This project uses a **client-server architecture**:
- The **frontend** is a lightweight HTML/JavaScript page for image upload and display.  
- The **backend** is a Python **Flask server** that uses a pre-trained **VGG19 model** in TensorFlow to apply style transfer.  

---

## 🚀 Features 
- Upload a **content image** (photo).  
- Upload a **style image** (painting).  
- Generate an **artistic masterpiece** combining both.  
- Simple **browser-based frontend**.  
- Backend runs with **ngrok** for public access.  

---

---

## ⚙️ How It Works
1. **Frontend (frontend.html)**  
   - User uploads a **content photo** and a **style painting**.  
   - Browser sends these images to the backend server.  

2. **Backend (app.py)**  
   - Flask receives the images.  
   - Pre-trained **VGG19 model** extracts **content and style features**.  
   - An optimization loop generates a new image that minimizes:  
     - **Content loss** (similarity to photo).  
     - **Style loss** (similarity to painting).  

3. **API Response**  
   - Backend returns the final **stylized image**.  
   - Frontend displays the masterpiece to the user.  

---

## 🖥️ Installation & Setup


```bash
git clone https://github.com/your-username/ai-artist-web-app.git
cd ai-artist-web-app


The backend performs heavy computation and is best run in an environment with a GPU for acceptable performance.

Get an ngrok Authtoken: The app.py script uses ngrok to create a public URL for the server. You will need a free account and an authtoken.

Sign up at https://dashboard.ngrok.com/signup.

Find your authtoken at https://dashboard.ngrok.com/get-started/your-authtoken.

Configure app.py: Open the app.py script in an editor.

Add Your Authtoken: Paste your ngrok authtoken into the NGROK_AUTH_TOKEN variable at the top of the script.

Execute the Script: Run the app.py script from your terminal:
