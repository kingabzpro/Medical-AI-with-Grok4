import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import gradio as gr
from firecrawl import FirecrawlApp, ScrapeOptions
from PIL import Image
from xai_sdk import Client
from xai_sdk.chat import image, system, tool, tool_result, user

# Initialize clients with connection pooling
client = Client(
    api_key=os.getenv("XAI_API_KEY"),
    timeout=3600,
    api_host="eu-west-1.api.x.ai",
)
fc = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


def get_medicine_info_fast(name: str) -> Dict:
    """Optimized medicine info fetcher with error handling"""
    try:
        results = fc.search(
            query=f"{name} medicine price availability",
            limit=1,
            scrape_options=ScrapeOptions(formats=["markdown"]),
        )
        snippet = results.data[0] if results.data else {}
        return {
            "name": name,
            "info_markdown": snippet.get("markdown", "N/A"),
            "url": snippet.get("url", "N/A"),
            "description": snippet.get("description", "N/A"),
            "status": "success",
        }
    except Exception as e:
        return {
            "name": name,
            "info_markdown": "Error fetching data",
            "url": "N/A",
            "description": f"Error: {str(e)}",
            "status": "error",
        }


def get_multiple_medicines_concurrent(
    medicine_names: List[str], max_workers: int = 5
) -> List[Dict]:
    """Fetch multiple medicine info concurrently"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_medicine = {
            executor.submit(get_medicine_info_fast, name): name
            for name in medicine_names
        }
        for future in as_completed(future_to_medicine):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                medicine_name = future_to_medicine[future]
                results.append(
                    {
                        "name": medicine_name,
                        "info_markdown": "Timeout or error",
                        "url": "N/A",
                        "description": f"Error: {str(e)}",
                        "status": "error",
                    }
                )
    return results


# Tool definitions
tool_definitions = [
    tool(
        name="get_medicine_info_fast",
        description="Fetch markdown info, URL, and description for a medicine via Firecrawl (optimized)",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the medicine"},
            },
            "required": ["name"],
        },
    ),
    tool(
        name="get_multiple_medicines_concurrent",
        description="Fetch info for multiple medicines concurrently",
        parameters={
            "type": "object",
            "properties": {
                "medicine_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of medicine names",
                },
                "max_workers": {
                    "type": "integer",
                    "description": "Maximum concurrent workers (default: 5)",
                    "default": 5,
                },
            },
            "required": ["medicine_names"],
        },
    ),
]

tools_map = {
    "get_medicine_info_fast": get_medicine_info_fast,
    "get_multiple_medicines_concurrent": get_multiple_medicines_concurrent,
}


def encode_image_from_bytes(image_bytes) -> str:
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_mime_type(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.format == "JPEG":
            return "image/jpeg"
        elif img.format == "PNG":
            return "image/png"
        else:
            return None
    except Exception:
        return None


def analyze_prescription_streaming(file_bytes):
    """Analyze prescription with streaming progress updates using generator"""
    try:
        yield "🔍 **Starting Analysis...**\n\nValidating uploaded image..."
        
        image_bytes = file_bytes
        mime_type = get_image_mime_type(image_bytes)
        if mime_type is None:
            yield "Error: Uploaded file is not a valid JPG or PNG image."
            return
        
        yield "✅ **Image Validated**\n\nPreparing image for AI analysis..."
            
        encoded_img = encode_image_from_bytes(image_bytes)
        image_data_url = f"data:{mime_type};base64,{encoded_img}"

        yield "🤖 **Connecting to Grok-4 AI**\n\nInitializing chat session..."
            
        # Create chat session
        chat = client.chat.create(
            model="grok-4",
            tools=tool_definitions,
            tool_choice="auto",
        )

        # Enhanced system prompt for better extraction
        chat.append(
            system(
                "You are MedGuide AI. Extract ALL medicine names from the prescription image. "
                "If you find multiple medicines, use get_multiple_medicines_concurrent to fetch "
                "all information at once for faster processing. For single medicine, use get_medicine_info_fast. "
                "Create a comprehensive markdown report."
            )
        )

        # User provides the prescription image
        chat.append(
            user(
                "Extract all medicine names from this prescription and get their details efficiently.",
                image(image_url=image_data_url, detail="high"),
            )
        )

        yield "🧠 **AI Processing Image**\n\nExtracting medicine names from prescription..."
            
        # Initial model call
        response = chat.sample()
        chat.append(response)
        
        # Show initial AI response
        yield f"✨ **Initial AI Response:**\n\n{response.content}\n\n---\n"

        # Execute tool calls if any
        if response.tool_calls:
            yield f"✨ **Initial AI Response:**\n\n{response.content}\n\n---\n\n🛠️ **Tool Calls Detected**\n\nExecuting API calls to fetch medicine information..."
                
            tool_results = []
            for i, tc in enumerate(response.tool_calls, 1):
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                
                args_display = json.dumps(func_args, indent=2)
                current_progress = f"✨ **Initial AI Response:**\n\n{response.content}\n\n---\n\n🛠️ **Tool Call {i}/{len(response.tool_calls)}:**\n\n**Function:** `{func_name}`\n\n**Arguments:**\n```json\n{args_display}\n```\n\n⏳ Fetching data from Firecrawl API..."
                yield current_progress
                
                result = tools_map[func_name](**func_args)
                tool_results.append((func_name, func_args, result))
                chat.append(tool_result(json.dumps(result)))
                
                # Show API response
                result_preview = json.dumps(result, indent=2)[:500] + "..." if len(json.dumps(result)) > 500 else json.dumps(result, indent=2)
                current_progress = f"✨ **Initial AI Response:**\n\n{response.content}\n\n---\n\n🛠️ **Tool Call {i}/{len(response.tool_calls)}:**\n\n**Function:** `{func_name}`\n\n**Arguments:**\n```json\n{args_display}\n```\n\n✅ **Firecrawl API Response:**\n```json\n{result_preview}\n```\n\n"
                yield current_progress

            all_tools_summary = ""
            for i, (func_name, func_args, result) in enumerate(tool_results, 1):
                all_tools_summary += f"\n**Tool Call {i}:**\n- Function: `{func_name}`\n- Arguments: {json.dumps(func_args)}\n- Status: ✅ Complete\n"
            
            yield f"✨ **Initial AI Response:**\n\n{response.content}\n\n---\n\n🛠️ **All Tool Calls Completed:**\n{all_tools_summary}\n---\n\n📝 **Generating Final Report**\n\nAI is creating comprehensive markdown report..."

            # Request final formatted report
            chat.append(
                user(
                    "Create a comprehensive markdown report with H2 heading for each medicine that contains: Description, "
                    "Typical Duration, Price Information, and Purchase Link"
                )
            )

            # Generate final report
            final = chat.sample()
            yield final.content
        else:
            yield response.content

    except Exception as e:
        yield f"Error analyzing prescription: {str(e)}"


# Gradio interface (Blocks version)
def main():
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown(
            """
            # MedGuide AI: Prescription Analyzer
            Upload a photo of your medical prescription. The AI will extract all medicine names and provide a detailed markdown report including description, duration, price, and purchase links.
            
            **🔄 Real-time Process Tracking:** Watch the AI work step-by-step below!
            """
        )
        with gr.Row():
            with gr.Column():
                file_input = gr.Image(label="Upload Prescription Image", type="pil")
                analyze_btn = gr.Button("Analyze Prescription", variant="primary")
            with gr.Column():
                report_output = gr.Markdown(
                    label="📋 Final Report",
                )
                time_output = gr.Markdown(label="⏱️ Processing Time")
                
                # Collapsible logs section
                with gr.Accordion("🔍 Processing Logs (Click to view details)", open=False):
                    logs_output = gr.Markdown(
                        label="Processing Details",
                        value="Logs will appear here during processing..."
                    )

        def analyze_with_streaming_progress(image):
            """Analyze prescription with streaming progress updates"""
            start_time = time.time()
            
            if image is None:
                yield "❌ **Error:** Please upload an image first.", "No image provided.", "Logs will appear here during processing..."
                return
            
            # Convert PIL image to bytes
            buf = io.BytesIO()
            image.save(
                buf,
                format=image.format if hasattr(image, "format") and image.format else "PNG",
            )
            image_bytes = buf.getvalue()
            
            # Accumulate all logs to show complete process
            all_logs = []
            final_markdown_report = ""
            
            # Use the streaming generator
            for progress_update in analyze_prescription_streaming(image_bytes):
                elapsed = time.time() - start_time
                
                # Check if this looks like the final markdown report
                # (contains medicine headings, descriptions, or is longer content)
                if (progress_update.startswith("# ") or 
                    progress_update.startswith("## ") or 
                    "Medicine" in progress_update and len(progress_update) > 100 or
                    "Description" in progress_update or
                    "Price" in progress_update or
                    "Duration" in progress_update):
                    # This is the final report - show in main report, keep logs in accordion
                    final_markdown_report = progress_update
                    final_logs = "\n\n".join(all_logs) if all_logs else "Processing completed successfully!"
                    yield final_markdown_report, f"✅ Completed in {elapsed:.2f} seconds", final_logs
                    return
                else:
                    # This is a process log - accumulate and show in logs section
                    all_logs.append(progress_update)
                    # Show processing message in main area, detailed logs in accordion
                    processing_message = f"🔄 **Processing in progress...**\n\nAnalyzing prescription image and fetching medicine information.\n\n*Check the Processing Logs section below for detailed step-by-step progress.*"
                    yield processing_message, f"Processing... {elapsed:.1f}s elapsed", "\n\n".join(all_logs)
            
            # If we somehow don't detect the final report, show the last update
            if not final_markdown_report:
                elapsed = time.time() - start_time
                final_logs = "\n\n".join(all_logs) if all_logs else "Processing completed."
                # Check if the last update could be the report
                if all_logs and len(all_logs[-1]) > 50:
                    yield all_logs[-1], f"✅ Completed in {elapsed:.2f} seconds", final_logs
                else:
                    yield "Analysis completed. Please check the processing logs for details.", f"✅ Completed in {elapsed:.2f} seconds", final_logs

        def show_initial_processing(image):
            """Show initial processing message"""
            if image is None:
                return "❌ **Error:** Please upload an image first.", "No image provided.", "Logs will appear here during processing..."
            return "🚀 **Processing Started!**\n\nInitializing analysis...", "Processing...", "Initializing..."

        # Event handlers
        analyze_btn.click(
            show_initial_processing,
            inputs=[file_input],
            outputs=[report_output, time_output, logs_output],
            queue=False,
        )
        
        analyze_btn.click(
            analyze_with_streaming_progress,
            inputs=[file_input],
            outputs=[report_output, time_output, logs_output],
            queue=True,
        )
        
        # Medical Disclaimer Section (Collapsible)
        with gr.Accordion("⚠️ Medical Disclaimer - Please Read", open=False):
            gr.Markdown(
                """
                **IMPORTANT NOTICE:** This application is for **informational purposes only** and should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.
                
                ### 🚨 Key Points:
                - **Always consult your doctor** or qualified healthcare provider with any questions about medications
                - **Verify all information** with your pharmacist or healthcare provider before taking any medication
                - **Do not rely solely** on this AI analysis for medical decisions
                - **AI may make errors** - always double-check medicine names, dosages, and instructions
                - **Emergency situations** require immediate medical attention, not AI analysis
                
                ### 📋 This Tool:
                - Provides general information about medicines found in prescription images
                - Uses AI to extract text and search for publicly available information
                - May not be 100% accurate in reading prescriptions or providing information
                - Should be used as a **supplementary tool only**
                
                ### 🩺 Always Remember:
                **Your healthcare provider knows your medical history and current condition best. When in doubt, always consult with them directly.**
                
                ---
                
                *By using this application, you acknowledge that you understand these limitations and agree to use the information responsibly.*
                """
            )

    demo.launch()


if __name__ == "__main__":
    main()
