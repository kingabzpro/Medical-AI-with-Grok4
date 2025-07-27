# MedGuide AI: Prescription Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-5.0+-green.svg)](https://gradio.app/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

A powerful AI-driven medical prescription analyzer that extracts medicine information from prescription images and provides comprehensive reports including prices, availability, dosage, and purchase links. Features a modern, clean interface with real-time streaming progress and collapsible sections for optimal user experience.


https://github.com/user-attachments/assets/20590d86-04ba-483b-b5b3-0b3fef4a9e3c


## ✨ Key Features

### 🔍 **Advanced AI Analysis**
- **Smart OCR**: Extract medicine names from prescription images using Grok-4 AI
- **Concurrent Processing**: Fetch multiple medicine details simultaneously for faster results
- **Real-time Streaming**: Watch AI work step-by-step with live progress updates
- **Optimized Performance**: Ultra-fast medicine lookup with aggressive optimization and fallback handling

### 🎨 **Modern User Interface**
- **Clean Design**: Professional Gradio web interface with intuitive layout
- **Smart Button States**: Button automatically disables during processing to prevent duplicate requests
- **Collapsible Logs**: Optional detailed processing logs hidden by default for clean UX
- **Collapsible Disclaimer**: Important medical warnings in expandable section
- **Responsive Layout**: Optimized for both desktop and mobile viewing

### 📋 **Comprehensive Reports**
- **Live Report Streaming**: Watch AI generate your medical report in real-time with streaming text updates
- **Structured Information**: Detailed markdown reports with medicine descriptions
- **Price & Availability**: Real-time pricing and purchase information
- **Processing Transparency**: Full visibility into AI decision-making process
- **Error Handling**: Robust error handling with informative feedback

## 🛠️ Technology Stack

- **AI Model**: Grok-4 by xAI for image analysis and text extraction
- **Web Framework**: Gradio for the user interface
- **Data Source**: Firecrawl API for medicine information retrieval
- **Image Processing**: PIL (Python Imaging Library)
- **Concurrency**: ThreadPoolExecutor for parallel processing

## 📋 Prerequisites

- Python 3.8 or higher
- xAI API key (for Grok-4 access)
- Firecrawl API key (for medicine data retrieval)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/kingabzpro/Medical-AI-with-Grok4.git
cd Medical-AI-with-Grok4
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example environment file and add your API keys:

Create the `.env` file with your actual API keys:

```env
XAI_API_KEY=your_xai_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

### 4. Run the Application

```bash
python app.py
```

The application will start and be available at `http://localhost:7860`

## 🎯 How to Use

### Basic Workflow
1. **Upload Image**: Click on the "Upload Prescription Image" area and select your prescription photo
2. **Analyze**: Click the "Analyze Prescription" button (button automatically changes to "⏳ Processing..." and disables to prevent duplicate requests)
3. **Monitor Progress**: Watch real-time processing updates in the main area and detailed logs in the collapsible section
4. **Live Streaming**: Experience real-time report generation as the AI streams the final medical report text
5. **View Results**: Get comprehensive medicine information including:
   - Medicine description and usage
   - Typical treatment duration
   - Price information and availability
   - Purchase links and sources
   - Total processing time

### 🗂️ Interface Features

#### **Main Report Area**
- Shows processing status during analysis
- Displays clean, formatted final report
- No technical clutter for better readability

#### **Processing Logs (Collapsible)**
- Click "🔍 Processing Logs" to expand detailed technical information
- View step-by-step AI analysis process
- Monitor API calls and responses
- Useful for debugging and transparency
- Hidden by default to keep interface clean

#### **Medical Disclaimer (Collapsible)**
- Click "⚠️ Medical Disclaimer" to expand important safety information
- Contains crucial warnings about AI limitations
- Emphasizes need for professional medical consultation
- Legal and safety guidelines for responsible use

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Gradio UI     │──▶│   Grok-4 AI     │───▶│  Firecrawl API   │
│   (Frontend)    │    │  (Agent & VLM)  │    │ (Search & Scrape)│
└─────────────────┘    └─────────────────┘    └──────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         └──────────────│  Python Backend │◀─────────────┘
                        │ (Coordination)  │
                        └─────────────────┘
```

## 📁 Project Structure

```
Medical-AI-with-Grok4/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .env                  # Your environment variables (not in git)
├── README.md             # Project documentation
└── .gitignore           # Git ignore file
```

## 🚨 Important Notes

- **Medical Disclaimer**: This tool is for informational purposes only and should not replace professional medical advice
- **Privacy**: Images are processed locally and not stored permanently
- **API Limits**: Be aware of rate limits for both xAI and Firecrawl APIs
- **Accuracy**: AI extraction may not be 100% accurate; always verify medicine information


